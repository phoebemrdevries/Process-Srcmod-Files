from __future__ import division
import math
import code
import datetime
import urllib
import utm
import os.path
import numpy as np
from scipy import io as sio
from okada_wrapper import dc3d0wrapper, dc3dwrapper
from shapely.ops import cascaded_union
from shapely.ops import unary_union
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pdb as check
from descartes import PolygonPatch
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import Point
import pandas as pd
from geopy.distance import great_circle

def getGridIdNames(xBufferFillGridVec, yBufferFillGridVec, zBufferFillGridVec):
            """ Flatten arrays of centroid points and get unique grid_id for each one
            """
            xBufferFillGridVecFlat = []
            yBufferFillGridVecFlat = []
            zBufferFillGridVecFlat = []
            gridIdFlat = []
            grid_id = range(len(xBufferFillGridVec))
            for iz in range(0, len(zBufferFillGridVec)):
                xBufferFillGridVecFlat.extend(xBufferFillGridVec)
                yBufferFillGridVecFlat.extend(yBufferFillGridVec)
                zBufferFillGridVecFlat.extend(zBufferFillGridVec[iz]*np.ones(np.size(xBufferFillGridVec)))
                gridIdFlat.extend([str(-zBufferFillGridVec[iz]) + '_' + str(id) for id in grid_id])
            return gridIdFlat, xBufferFillGridVecFlat, yBufferFillGridVecFlat, zBufferFillGridVecFlat


def RotateCoords(x, y, x_offset, y_offset, angle):
        """Rotate x and y observation coordinates in a local reference frame.
        
        Rotate a single set of x and y observation coordinates into a local
        reference frame appropriate for okada_wrapper call.  The two steps are: 1)
        calculate x and y coordinates relative to fault coordinates system with a
        translation and 2) rotate to correct for strike (already converted to a
        Cartesian angle).
        
        Inputs:
        x: x-coordinate to rotate
        y: y-coordinate to rotate
        x_offset: x-coordinate of local fault cetered coordinate system
        y_offset: y-coordinate of local fault cetered coordinate system
        angle: Angle (Cartesian, not strike) to rotate coordinates by (degrees).
        
        Returns:
        x and y coordinates rotated into a local reference frame.
        """
        x_local = x - x_offset
        y_local = y - y_offset
        angle = np.radians(1.0 * angle)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]]) #counterclockwise
        x_rot_vec = np.dot(rot_matrix, np.array([x_local, y_local]))
        return x_rot_vec[0], x_rot_vec[1]


def RotateDisplacements(u, angle):
        """Rotate uxprime and uyprime out of local reference frame.
        Inputs:
        uxprime: ux to rotate
        uyprime: uy to rotate
        angle: Angle (Cartesian, not strike) to rotate coordinates by (degrees).
        
        Returns:
        ux and uy in global x,y reference frame.
        """
        angle = np.radians(1.0 * angle)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]]) #counterclockwise
        d_rot_vec = np.dot(rot_matrix, np.array([u[0], u[1]]))
        return [d_rot_vec[0], d_rot_vec[1], u[2]]


def RotateTensor(tensor, angle):
        """Rotate tensor out of local reference frame.
        Inputs:
        tensor: tensor to rotate
        angle: Angle (Cartesian, not strike) to rotate coordinates by (degrees).
        
        Returns:
        rotated tensor
        """
        angle = np.radians(1.0 * angle)
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) #counterclockwise
        T_rot = np.dot(np.dot(rot_matrix, tensor), rot_matrix.T)
        return T_rot

def calcOkadaDisplacementStress(x, y, z, event_srcmod, lambda_lame, mu_lame):
        """Calculate strains and stresses from SRCMOD event with Okada (1992).
        Calculate nine element symmetric elastic strain and stress tensors at
        observation coordinates using Okada (1992).  Dislocation parameters are
        given an event_srcmod dictionary which contains the geometry and slip
        distribution for a given SRCMOD event.
        Inputs:
        x: List of x-coordinates of observation points (meters)
        y: List of y-coordinates of observation points (meters)
        z: List of z-coordinates of observation points (meters, negative down)
        event_srcmod: Dictionary with SRCMOD event parameters for one event
        lambda_lame: Lame's first parameter (Pascals)
        mu_lame: Lame's second parameter, shear modulus (Pascals)
        Returns:
        strains, stresses: Lists of 3x3 numpy arrays with full strain
        and stress tensors at each 3d set of obervation coordinates
        """
        strains = []
        stresses = []
        displacements = []
        mindistance = []
        
        # Define the material parameter that Okada's Greens functions is sensitive too
        alpha = (lambda_lame + mu_lame) / (lambda_lame + 2 * mu_lame)
        
        for j in range(len(x)):
            strain = np.zeros((3, 3))
            stress = np.zeros((3, 3))
            displacement = np.zeros([3])
            distance = []
            for i in range(len(event_srcmod['x1'])):

                # Translate and (un)rotate observation coordinates to get a local reference frame in which top edge of fault is aligned with x-axis
                x_rot, y_rot = RotateCoords(x[j], y[j], event_srcmod['x1Utm'][i], event_srcmod['y1Utm'][i], -1.0 * event_srcmod['angle'][i])
                # get rotated fault coordinates (otherwise fault patch might be offset on x-axis)
                x2_f, y2_f = RotateCoords(event_srcmod['x2Utm'][i], event_srcmod['y2Utm'][i], event_srcmod['x1Utm'][i], event_srcmod['y1Utm'][i], -1.0 * event_srcmod['angle'][i])
                x_fault1 = np.min([0., x2_f])
                x_fault2 = np.max([0., x2_f])
                assert (x_fault2-x_fault1)-event_srcmod['length'][i] < 100
                # Calculate elastic deformation using Okada 1992 (BSSA)
                # Seven arguments to DC3DWrapper are required:
                # alpha = (lambda + mu) / (lambda + 2 * mu)
                # xo = 3-vector representing the observation point (x, y, z in the original)
                # depth = the depth of the fault origin
                # dip = the dip-angle of the rectangular dislocation surface
                # strike_width = the along-strike range of the surface (al1,al2 in the original)
                # dip_width = the along-dip range of the surface (aw1, aw2 in the original)
                # dislocation = 3-vector representing the direction of motion on the surface (DISL1, DISL2, DISL3)
                success, uvec, gradient_tensor = dc3dwrapper(alpha,
                                                             [x_rot[0], y_rot[0], z[j]], #observation depth has to be negative
                                                             event_srcmod['z1'][i],
                                                      event_srcmod['dip'][i],
                                                             [x_fault1, x_fault2],
                                                             [-1.0*event_srcmod['width'][i], 0],
                                                      [event_srcmod['slipStrike'][i],
                                                       event_srcmod['slipDip'][i],
                                                       0.0])
                # Tensor algebra definition of strain
                cur_straintmp = 0.5 * (gradient_tensor.T + gradient_tensor)
                #
                cur_strain = RotateTensor(cur_straintmp, 1.0 * event_srcmod['angle'][i])
                strain += cur_strain
                # Tensor algebra constituitive relationship for elasticity
                stress += (lambda_lame * np.eye(cur_strain.shape[0]) * np.trace(cur_strain) + 2. * mu_lame * cur_strain)
                displacement += RotateDisplacements(uvec, 1.0 * event_srcmod['angle'][i])
                distance.append(np.sqrt(np.power(x_rot[0],2.) + np.power(y_rot[0],2.)))
            
            mindistance.append(np.min(np.array(distance)))
            strains.append(strain)
            stresses.append(stress)
            displacements.append(displacement)
        return displacements, strains, stresses, mindistance


def calcFaultBuffer(EventSrcmod, distance):
    # Create buffer around fault with shapely
    circles = (Point(EventSrcmod['x1Utm'][0], EventSrcmod['y1Utm'][0]).buffer(distance))
    circlesall = []
    for iPatch in range(0, len(EventSrcmod['x1'])): # Plot the edges of each fault patch fault patches
        circlestmp = (Point(EventSrcmod['x1Utm'][iPatch], EventSrcmod['y1Utm'][iPatch]).buffer(distance))
        circles = circles.union(circlestmp)
        circlesall.append(Point(EventSrcmod['x1Utm'][iPatch], EventSrcmod['y1Utm'][iPatch]).buffer(distance))
        circlestmp = (Point(EventSrcmod['x2Utm'][iPatch], EventSrcmod['y2Utm'][iPatch]).buffer(distance))
        circles = circles.union(circlestmp)
        circlesall.append(Point(EventSrcmod['x2Utm'][iPatch], EventSrcmod['y2Utm'][iPatch]).buffer(distance))
        circlestmp = (Point(EventSrcmod['x3Utm'][iPatch], EventSrcmod['y3Utm'][iPatch]).buffer(distance))
        circles = circles.union(circlestmp)
        circlesall.append(Point(EventSrcmod['x3Utm'][iPatch], EventSrcmod['y3Utm'][iPatch]).buffer(distance))
        circlestmp = (Point(EventSrcmod['x4Utm'][iPatch], EventSrcmod['y4Utm'][iPatch]).buffer(distance))
        circles = circles.union(circlestmp)
        circlesall.append(Point(EventSrcmod['x4Utm'][iPatch], EventSrcmod['y4Utm'][iPatch]).buffer(distance))
    polygonBuffer = circles
    temp = np.array(polygonBuffer.exterior).flatten()
    xBuffer = temp[0::2]
    yBuffer = temp[1::2]
    return(xBuffer, yBuffer, polygonBuffer)


def calcBufferGridPoints(xBuffer, yBuffer, polygonBuffer, spacingGrid):
    xBufferFillVec = np.arange(np.min(xBuffer), np.max(xBuffer), spacingGrid)
    yBufferFillVec = np.arange(np.min(yBuffer), np.max(yBuffer), spacingGrid)
    xBufferFillGrid, yBufferFillGrid = np.meshgrid(xBufferFillVec, yBufferFillVec)
    gridcells = []
    XCentroidPts = []
    YCentroidPts = []
    for i in range(0,len(yBufferFillVec)-1):
        for jj in range(0, len(xBufferFillVec)-1):
            p1 = [xBufferFillGrid[i,jj], yBufferFillGrid[i,jj]]
            p2 = [xBufferFillGrid[i+1,jj], yBufferFillGrid[i+1,jj]]
            p3 = [xBufferFillGrid[i+1,jj+1], yBufferFillGrid[i+1,jj+1]]
            p4 = [xBufferFillGrid[i, jj+1], yBufferFillGrid[i, jj+1]]
            gridcells.append(Polygon([p1, p2, p3, p4]))
            XCentroidPts.append(xBufferFillGrid[i,jj] + (xBufferFillGrid[i,jj+1]-xBufferFillGrid[i,jj])/2.)
            YCentroidPts.append(yBufferFillGrid[i,jj] + (yBufferFillGrid[i+1, jj]-yBufferFillGrid[i, jj])/2.)
    # Select only those grid points inside of buffered region
    xBufferFillGridVec = xBufferFillGrid.flatten()
    yBufferFillGridVec = yBufferFillGrid.flatten()
    deleteIdx = []
    for iCell in range(0, len(gridcells)):
        isIn = polygonBuffer.contains(gridcells[iCell])
        if isIn == False:
            deleteIdx.append(iCell)
    gridcellsInBuffer = [v for i,v in enumerate(gridcells) if i not in deleteIdx]
    XCentroidPtsInBuffer = [v for i,v in enumerate(XCentroidPts) if i not in deleteIdx]
    YCentroidPtsInBuffer = [v for i,v in enumerate(YCentroidPts) if i not in deleteIdx]
    return(np.array(XCentroidPtsInBuffer), np.array(YCentroidPtsInBuffer), gridcellsInBuffer)


def StressMasterCalc(EventSrcmod, StressTensorBuffer, StrainTensorBuffer, DisplacementVectorBuffer, coefficientOfFriction):
    """Function calculates all CFS, max shear, etc. Assembles dictionary BigBig to be written to CSV."""

    Cfs = dict()
    Cfs['strikeMean'] = EventSrcmod['strikeMean']
    Cfs['dipMean'] = EventSrcmod['dipMean']
    Cfs['rakeMean'] = EventSrcmod['rakeMean']
    Cfs['nVecInPlane'], Cfs['nVecNormal'] = cfsVectorsFromAzimuth(Cfs['strikeMean'], Cfs['dipMean'], Cfs['rakeMean'])

    BigBig = dict()
    
    #full stress matrix first
    BigBig['stresses_full_cfs']  = calcCfs(StressTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['stresses_full_cfs_shear_only'] = calcShearStress(StressTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'])
    BigBig['stresses_full_cfs_normal'] = calcNormalStress(StressTensorBuffer, Cfs['nVecNormal'], coefficientOfFriction)
    BigBig['stresses_full_cfs_total_shear'] = CfsTotalShear(StressTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['stresses_full_cfs_total'] = calcCfsTotal(StressTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['stresses_full_max_shear'] = MaximumShear(StressTensorBuffer)
    BigBig['stresses_full_i1'], BigBig['stresses_full_i2'], BigBig['stresses_full_i3'] = TensorInvariants(StressTensorBuffer)
    BigBig['stresses_full_eigmax'], BigBig['stresses_full_eigmed'], BigBig['stresses_full_eigmin'] = Eigenvalues(StressTensorBuffer)

    # now raw full stresses
    BigBig['stresses_full_xx'], BigBig['stresses_full_xy'], BigBig['stresses_full_xz'], BigBig['stresses_full_yy'], BigBig['stresses_full_yz'],BigBig['stresses_full_zz'] = ExtractStressComponents(StressTensorBuffer)

    #full strain matrix next
    BigBig['strains_full_cfs']  = calcCfs(StrainTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['strains_full_cfs_shear_only'] = calcShearStress(StrainTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'])
    BigBig['strains_full_cfs_normal'] = calcNormalStress(StrainTensorBuffer, Cfs['nVecNormal'], coefficientOfFriction)
    BigBig['strains_full_cfs_total_shear'] = CfsTotalShear(StrainTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['strains_full_cfs_total'] = calcCfsTotal(StrainTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['strains_full_max_shear'] = MaximumShear(StrainTensorBuffer)
    BigBig['strains_full_i1'], BigBig['strains_full_i2'], BigBig['strains_full_i3'] = TensorInvariants(StrainTensorBuffer)

    BigBig['strains_full_eigmax'], BigBig['strains_full_eigmed'], BigBig['strains_full_eigmin'] = Eigenvalues(StrainTensorBuffer)

    # now raw full strains
    BigBig['strains_full_xx'], BigBig['strains_full_xy'], BigBig['strains_full_xz'], BigBig['strains_full_yy'], BigBig['strains_full_yz'],BigBig['strains_full_zz'] = ExtractStressComponents(StrainTensorBuffer)
    
    # now deviatoric stress
    DeviatoricStressTensorBuffer = DeviatoricTensor(StressTensorBuffer)
    BigBig['stresses_deviatoric_cfs']  = calcCfs(DeviatoricStressTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['stresses_deviatoric_cfs_shear_only'] = calcShearStress(DeviatoricStressTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'])
    BigBig['stresses_deviatoric_cfs_normal'] = calcNormalStress(DeviatoricStressTensorBuffer, Cfs['nVecNormal'], coefficientOfFriction)
    BigBig['stresses_deviatoric_cfs_total_shear'] = CfsTotalShear(DeviatoricStressTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['stresses_deviatoric_cfs_total'] = calcCfsTotal(DeviatoricStressTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['stresses_deviatoric_max_shear'] = MaximumShear(DeviatoricStressTensorBuffer)
    BigBig['stresses_deviatoric_i1'], BigBig['stresses_deviatoric_i2'], BigBig['stresses_deviatoric_i3'] = TensorInvariants(DeviatoricStressTensorBuffer)

    BigBig['stresses_deviatoric_eigmax'], BigBig['stresses_deviatoric_eigmed'], BigBig['stresses_deviatoric_eigmin'] = Eigenvalues(DeviatoricStressTensorBuffer)

    # now raw deviatoric stresses
    BigBig['stresses_deviatoric_xx'], BigBig['stresses_deviatoric_xy'], BigBig['stresses_deviatoric_xz'], BigBig['stresses_deviatoric_yy'], BigBig['stresses_deviatoric_yz'],BigBig['stresses_deviatoric_zz'] = ExtractStressComponents(DeviatoricStressTensorBuffer)

    # and deviatoric strain
    DeviatoricStrainTensorBuffer = DeviatoricTensor(StrainTensorBuffer)
    BigBig['strains_deviatoric_cfs']  = calcCfs(DeviatoricStrainTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['strains_deviatoric_cfs_shear_only'] = calcShearStress(DeviatoricStrainTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'])
    BigBig['strains_deviatoric_cfs_normal'] = calcNormalStress(DeviatoricStrainTensorBuffer, Cfs['nVecNormal'], coefficientOfFriction)
    BigBig['strains_deviatoric_cfs_total_shear'] = CfsTotalShear(DeviatoricStrainTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['strains_deviatoric_cfs_total'] = calcCfsTotal(DeviatoricStrainTensorBuffer, Cfs['nVecNormal'], Cfs['nVecInPlane'], coefficientOfFriction)
    BigBig['strains_deviatoric_max_shear'] = MaximumShear(DeviatoricStrainTensorBuffer)
    BigBig['strains_deviatoric_i1'], BigBig['strains_deviatoric_i2'], BigBig['strains_deviatoric_i3'] = TensorInvariants(DeviatoricStrainTensorBuffer)

    BigBig['strains_deviatoric_eigmax'], BigBig['strains_deviatoric_eigmed'], BigBig['strains_deviatoric_eigmin'] = Eigenvalues(DeviatoricStrainTensorBuffer)

    # now raw deviatoric strains
    BigBig['strains_deviatoric_xx'], BigBig['strains_deviatoric_xy'], BigBig['strains_deviatoric_xz'], BigBig['strains_deviatoric_yy'], BigBig['strains_deviatoric_yz'],BigBig['strains_deviatoric_zz'] = ExtractStressComponents(DeviatoricStrainTensorBuffer)

    # Add displacements
    BigBig['ux'], BigBig['uy'], BigBig['uz'] = ExtractDisplacementComponents(DisplacementVectorBuffer)
    
    return Cfs, BigBig

def ExtractStressComponents(tensors):
    xx = []
    xy = []
    xz = []
    yy = []
    yz = []
    zz = []
    for tensor in tensors:
        xx.append(tensor[0][0])
        xy.append(tensor[0][1])
        xz.append(tensor[0][2])
        yy.append(tensor[1][1])
        yz.append(tensor[1][2])
        zz.append(tensor[2][2])
    return xx, xy, xz, yy, yz, zz

def ExtractDisplacementComponents(displacements):
    ux = []
    uy = []
    uz = []
    for uvec in displacements:
        ux.append(uvec[0])
        uy.append(uvec[1])
        uz.append(uvec[2])
    return ux, uy, uz

def calcCfs(tensors, n_vec_normal, n_vec_in_plane, coefficient_of_friction):
    cfc1 = []
    for tensor in tensors:
      delta_tau = np.dot(np.dot(tensor, n_vec_normal), n_vec_in_plane)
      delta_sigma = np.dot(np.dot(tensor, n_vec_normal), n_vec_normal)
      cfc1.append(delta_tau + coefficient_of_friction * delta_sigma)
    return np.array(cfc1)

def calcNormalStress(tensors, n_vec_normal, coefficient_of_friction):
    cfc = []
    for tensor in tensors:
      delta_sigma = np.dot(np.dot(tensor, n_vec_normal), n_vec_normal)
      cfc.append(coefficient_of_friction*delta_sigma)
    return np.array(cfc)

def calcShearStress(tensors, n_vec_normal, n_vec_in_plane):
    cfc = []
    for tensor in tensors:
        delta_tau = np.dot(np.dot(tensor, n_vec_normal), n_vec_in_plane)
        cfc.append(delta_tau)
    return np.array(cfc)

def calcCfsTotal(tensors, n_vec_normal, n_vec_in_plane, coefficient_of_friction):
    """Calculate total shear Coulomb criteria."""
    cfc = []
    n_vec_cross = np.cross(n_vec_normal, n_vec_in_plane)
    for tensor in tensors:
      delta_tau1 = np.dot(np.dot(tensor, n_vec_normal), n_vec_in_plane)
      delta_tau2 = np.dot(np.dot(tensor, n_vec_normal), n_vec_cross)
      delta_sigma = np.dot(np.dot(tensor, n_vec_normal), n_vec_normal)
      cfc.append(np.abs(delta_tau1) + np.abs(delta_tau2)
               + coefficient_of_friction * delta_sigma)
    return np.array(cfc)

def CfsTotalShear(tensors, n_vec_normal, n_vec_in_plane, coefficient_of_friction):
    """Calculate total shear Coulomb criteria.
        We did not note absolute values here in Table 1 of Paper 1"""
    cfc = []
    n_vec_cross = np.cross(n_vec_normal, n_vec_in_plane)
    for tensor in tensors:
        delta_tau1 = np.dot(np.dot(tensor, n_vec_normal), n_vec_in_plane)
        delta_tau2 = np.dot(np.dot(tensor, n_vec_normal), n_vec_cross)
        cfc.append(np.abs(delta_tau1) + np.abs(delta_tau2))
    return np.array(cfc)

def getClockwiseRotationMatrixZaxis(angle):
    """Angle input is in radians"""
    Rot_matrix = np.array([[math.cos(angle),
                            math.sin(angle), 0.],
                           [-math.sin(angle),
                            math.cos(angle), 0.],
                           [0., 0., 1.]])
    return Rot_matrix

def getClockwiseRotationMatrixYaxis(angle):
    """Angle input is in radians"""
    Rot_matrix = np.array([[math.cos(angle), 0.,
                            -math.sin(angle)],
                           [0., 1., 0.],
                           [math.sin(angle),
                            0., math.cos(angle)]])
    return Rot_matrix

def cfsVectorsFromAzimuth(meanStrike, meanDip, meanRake):
    """Finds the CFS normal vectors.
        fault_azimuth: Degress of the fault azimuth.
        fault_dip: Degress of the fault.
        strikeSlip: slip in strike direction
        slipDip: slip in dip direction
        Returns:
        Tuple of rotated vectors.
        """
    R2 = getClockwiseRotationMatrixZaxis(math.radians(meanStrike))
    R1 = getClockwiseRotationMatrixYaxis(math.radians(90.-meanDip))
    R3 = getClockwiseRotationMatrixYaxis(math.radians(180.-meanDip))
    n_vec_normal = np.dot(R2, np.dot(R1, [1, 0, 0]))
    v_ss = np.dot(R2, [0, 1, 0]) 
    v_ds = np.dot(R2, np.dot(R3, [1, 0, 0]))
    R4 = np.array([[math.cos(math.radians(meanRake)), -math.sin(math.radians(meanRake))], [math.sin(math.radians(meanRake)), math.cos(math.radians(meanRake))]])
    [slipSS, dipSS] = np.dot(R4, [1., 0.])
    n_vec_in_plane = slipSS*v_ss + dipSS*v_ds
    return (n_vec_in_plane, n_vec_normal)


def MaximumShear(tensors):
        """ Calculate the maximum shear of a symmetric 3x3 tensor. This is just half the
        difference between the largest and smallest eigenvalues.
        tensors: A list of symmetric 3x3 arrays.
        Returns:
        ret: a list of maximum shear values.
        """
        ret = []
        for tensor in tensors:
            eigen_values = list(np.linalg.eigvalsh(tensor))
            ret.append((max(eigen_values) - min(eigen_values)) / 2.0)
        return np.array(ret)

def Eigenvalues(tensors):
        """ Calculate eigenvalues of matrix.
        """
        eigmax = []
        eigmed = []
        eigmin = []
        for tensor in tensors:
            eigen_values = list(np.linalg.eigvalsh(tensor))
            eigen_values.sort()
            eigmax.append(eigen_values[2])
            eigmed.append(eigen_values[1])
            eigmin.append(eigen_values[0])
        return np.array(eigmax), np.array(eigmed), np.array(eigmin)

def TensorInvariants(tensors):
        """ Calculate the invariants component of a symmetric 3x3 tensor. Forumulas are
        according following Malvern (1969) and conveniently replicated on Wikipedia:
        https://en.wikipedia.org/wiki/Cauchy_stress_tensor
        tensors: A list of symmetric 3x3 arrays.
        Returns:
        Three lists (i1, i2, i3) of numpy arrays.
        """
        i1 = []
        i2 = []
        i3 = []
        for tensor in tensors:
            i1.append(np.trace(tensor))
            i2tmp = (np.linalg.det(tensor[0:2, 0:2]) + np.linalg.det(np.double([[tensor[0,0], tensor[0,2]], [tensor[2,0], tensor[2,2]]])) + np.linalg.det(tensor[1:3, 1:3]))
            i2.append(i2tmp)
            i2tmpcheck = 0.5*(np.power(np.trace(tensor), 2.) - np.trace(np.matmul(tensor, tensor)))
            assert np.sign(i2tmpcheck) == np.sign(i2tmp)
            assert np.abs(i2tmpcheck)-0.001*np.abs(i2tmpcheck) < np.abs(i2tmp) < np.abs(i2tmpcheck)+0.001*np.abs(i2tmpcheck)
            i3.append(np.linalg.det(tensor))
        return np.array(i1), np.array(i2), np.array(i3)


def DeviatoricTensor(tensors):
        """ Calculate the deviatoric component of a 3x3 tensor. This done by subtracting
        out the isotropic part of the full tensor given by the average of the main
        diagonal elements. Operates on either a single 3x3 tensor or a list of 3x3
        arrays.
        tensors: A list of 3x3 arrays.
        Returns:
        A list of 3x3 arrays with the isotropic component subtracted out.
        """
        ret = []
        for tensor in tensors:
            avg = np.trace(tensor) / 3.
            ret.append(tensor - avg * np.eye(3))
        return np.array(ret)

