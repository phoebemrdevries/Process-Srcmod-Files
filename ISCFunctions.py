from __future__ import division
import datetime
import numpy as np
from shapely.geometry import Point
import pdb as check
import pandas as pd
import time

# Function to check to see if a string can be converted into a float.  Useful for error checking.
def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Function to either load ISC event catalog between two different dates
def getIscEventCatalog(startDateTime, endDateTime, EventSrcmod, catalogType):
    EpiLat = EventSrcmod['epicenterLatitude']
    EpiLon = EventSrcmod['epicenterLongitude']

    if catalogType == 'REVIEWED':
        filename = '../Data/isc_rev.pkl'
    elif catalogType == 'EHB':
        filename = '../Data/isc_ehb.pkl'
    elif catalogType == 'COMPREHENSIVE':
        filename = '../Data/isc_comp.pkl'

    ti = time.time()
    df = pd.read_pickle(filename)
    print 'Loading pickle took ' + str(time.time()-ti) + ' seconds.'

    #convert strings to numbers and fill in blanks with NaNs
    df[['magnitude', 'depth', 'latitude', 'longitude']]=df[['magnitude', 'depth', 'latitude', 'longitude']].apply(pd.to_numeric, errors='coerce')

    #quality filtering
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df[(df.magnitude.notnull()) & (df.depth.notnull()) & (df.latitude.notnull()) & (df.longitude.notnull()) & (df.magnitude_type.notnull()) & (df.magnitude_author.notnull())]

    #make sure UTM works
    df = df[(df.latitude < 84) & (df.latitude > -84)]
    df = df[(df['depth']<>0.0)]

    # time filtering
    df = df[(df.datetime>startDateTime) & (df.datetime<endDateTime)]
    df.replace('\s+', '',regex=True,inplace=True)

    #magnitude type filtering 
    df = df[df['magnitude_type'].isin(['Mb','mb','MB', 'mB', 'ML','Ml','ml','mL','Mw','MW','mW', 'mw', 'Ms','MS', 'mS', 'ms'])]

    #write to dict
    catalog = dict()
    catalog['latitude'] = list(df['latitude'])
    catalog['longitude'] = list(df['longitude'])
    catalog['magnitude'] = list(df['magnitude'])
    catalog['depth'] = list(np.abs(df['depth']))

    return(catalog)


def getNearFieldIscEventsBuffer(Catalog, EventSrcmod, polygonBuffer):
    # Convert longitude and latitudes to local UTM coordinates
    Catalog['xUtm'], Catalog['yUtm'] = EventSrcmod['projEpicenter'](Catalog['longitude'], Catalog['latitude'])
    # Determine whether or not the catalog events are withing the polygon buffer
    deleteIdx = []
    Catalog['distanceToEpicenter'] = []
    for iIsc in range(0, len(Catalog['xUtm'])):
        srcmodIscDistance = np.sqrt((Catalog['xUtm'][iIsc] - EventSrcmod['epicenterXUtm'])**2 + 
                                    (Catalog['yUtm'][iIsc] - EventSrcmod['epicenterYUtm'])**2)
        Catalog['distanceToEpicenter'].append(srcmodIscDistance)
        candidatePoint = Point(Catalog['xUtm'][iIsc], Catalog['yUtm'][iIsc])
        isIn = polygonBuffer.contains(candidatePoint)
        if isIn == False:
            deleteIdx.append(iIsc)
    # Remove all catalog earthquakes that are not in field of interest from lists in dict Catalog
    deleteIdx = sorted(deleteIdx, reverse=True)
    for iKey in Catalog.keys():
        for iDeleteIdx in range(0, len(deleteIdx)):
            del Catalog[iKey][deleteIdx[iDeleteIdx]]
    return(Catalog)


def binNearFieldIscEventsBuffer(Catalog, EventSrcmod, polygonBuffer, gridcellsInBuffer, zBuffer):
   Counts = []
   for iz in range(0, len(zBuffer)-1):
      for iCell in range(0, len(gridcellsInBuffer)):
         magnitudestmp = []
         timevectmp = []
         depthstmp = []
         momentstmp = []
         for iIsc in range(0, len(Catalog['xUtm'])):
             z = 1000.*Catalog['depth'][iIsc]
             candidatePoint = Point(Catalog['xUtm'][iIsc], Catalog['yUtm'][iIsc])
             isIn = ((gridcellsInBuffer[iCell].contains(candidatePoint)) and (z >= -1.*zBuffer[iz]) and (z < -1.*zBuffer[iz+1]))
             if isIn == True:
                 magnitudestmp.append(Catalog['magnitude'][iIsc])
         Counts.append(len(magnitudestmp))
   return Counts


