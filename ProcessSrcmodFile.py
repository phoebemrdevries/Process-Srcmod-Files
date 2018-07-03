from __future__ import division
import datetime
import numpy as np
import pdb as check
import Readsrcmod
import Calculate
import ISCFunctions
import WriteOutput
import time
import sys
import csv
import datetime

def ProcessSrcmod(fileName):
    print '\nStarting calculations for ' + fileName + '...\n'
    FSBFilesFolder = './' #path to folder with fsp file
    # Parameters for CFS calculation and visualization
    lambdaLame = 3e10 # First Lame parameter (Pascals)
    muLame = 3e10 # Second Lame parameter (shear modulus, Pascals)
    coefficientOfFriction = 0.4 # Coefficient of friction
    obsDepth = 0; # depth of observation coordinates just for disc visualization
    useUtm = True
    catalogType = 'REVIEWED'
    # Options are:
    # 'COMPREHENSIVE': most earthquakes. Not human reviewed
    # 'REVIEWED': slightly fewer earthquakes. Some human quality control
    # 'EHB': many fewer earthquakes. Human quality control and precise relocations
    captureDays = 365 # Consider ISC earthquakes for this many days after day of main shock
    nearFieldDistance = 100e3 # Keep only those ISC earthquakes withing this distance of the SRCMOD epicenter
    spacingGrid = 5e3
    maxDepth = 50000
    zBuffer = -1.*np.arange(0, maxDepth+spacingGrid, spacingGrid)
    zBufferFillGridVec = -1.*(np.arange(spacingGrid, maxDepth+spacingGrid, spacingGrid)-spacingGrid/2.)

    # get exact time of earthquake
    with open('MainShockTimes.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if (row['earthquake'][2:-10] == fileName[1:-4]):
                ExactStart = row['time']
                datetime_Exact = datetime.datetime.strptime(ExactStart[:-4], '%Y-%m-%d %H:%M:%S')

    print 'Reading in SRCMOD fault geometry and slip distribution for this representation of the event...\n'
    EventSrcmod = Readsrcmod.ReadSrcmodFile(fileName, FSBFilesFolder)

    print 'Calculating regular grid over region inside fault buffer...\n'
    xBuffer, yBuffer, polygonBuffer = Calculate.calcFaultBuffer(EventSrcmod, nearFieldDistance)
    
    print 'Calculating fault buffer grid points...\n'
    xBufferFillGridVec, yBufferFillGridVec, gridcellsInBuffer = Calculate.calcBufferGridPoints(xBuffer, yBuffer, polygonBuffer, spacingGrid)

    print 'Generating unique grid identifiers and flattening observation arrays...\n'
    gridIdFlat, xBufferFillGridVecFlat, yBufferFillGridVecFlat, zBufferFillGridVecFlat = Calculate.getGridIdNames(xBufferFillGridVec, yBufferFillGridVec, zBufferFillGridVec)

    ti = time.time()
    print 'Calculating displacement vectors and stress tensors at observation coordinates...'
    DisplacementVectorBuffer, StrainTensorBuffer, StressTensorBuffer, Distances = Calculate.calcOkadaDisplacementStress(xBufferFillGridVecFlat, yBufferFillGridVecFlat, zBufferFillGridVecFlat, EventSrcmod, lambdaLame, muLame)
    print 'Calculations took ' + str(time.time()-ti) + ' seconds.\n'

    print 'Resolving Coulomb failure stresses on receiver planes...\n'
    Cfs, BigBig = Calculate.StressMasterCalc(EventSrcmod, StressTensorBuffer, StrainTensorBuffer, DisplacementVectorBuffer, coefficientOfFriction)

    print 'Getting start/end dates...'
    startDateTime = EventSrcmod['datetime']
    datetime_ExactStart = datetime_Exact + datetime.timedelta(seconds=1)
    assert datetime_ExactStart - datetime.timedelta(hours=36) < startDateTime < datetime_ExactStart + datetime.timedelta(hours=36)
    endDateTime = datetime_ExactStart + datetime.timedelta(days=captureDays) - datetime.timedelta(seconds=1)
    
    ti = time.time()
    print 'Reading in ISC data for capture days after the date of the SRCMOD event...'
    Catalog = ISCFunctions.getIscEventCatalog(datetime_ExactStart, endDateTime, EventSrcmod, catalogType)
    print 'ISC retrieval and time/quality filtering took ' + str(time.time()-ti) + ' seconds in total.\n'

    print 'Events within the fault buffer...\n'
    Catalog = ISCFunctions.getNearFieldIscEventsBuffer(Catalog, EventSrcmod, polygonBuffer)
    
    ti = time.time()
    print 'Binning the events by grid cell and assembing magnitudes, times, counts, etc...\n'
    Counts = ISCFunctions.binNearFieldIscEventsBuffer(Catalog, EventSrcmod, polygonBuffer, gridcellsInBuffer, zBuffer)
    
    print 'Binning ISC events took ' + str(time.time()-ti) + ' seconds.\n'

    BigBig['x'] = xBufferFillGridVecFlat
    BigBig['y'] = yBufferFillGridVecFlat
    BigBig['z'] = zBufferFillGridVecFlat
    BigBig['grid_id'] = gridIdFlat
    BigBig['grid_aftershock_count'] = Counts

    print 'Writing output CSV file...\n'
    ti = time.time()
    fileNameReturn = WriteOutput.WriteCSV(fileName, BigBig)
    print 'Writing output took ' + str(time.time()-ti) + ' seconds.\n'

    return

if __name__ == '__main__':
    ProcessSrcmod(*sys.argv[1:])

