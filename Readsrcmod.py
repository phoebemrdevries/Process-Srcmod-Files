import collections
import datetime
import logging
import math
import re
from cmath import rect, phase
import numpy as np
import pyproj
import utm
import pdb as check
import copy

# Regular expressions that will parse the text Srcmod files.
# TAGS are of the form: 'xxx : yyy zzz'
#Note: lines 13-18 and 88-418 are based on https://github.com/google/stress_transfer/tree/master/stress_transfer
TAGS_RE = re.compile(r'(\w+\s*:\s*(?:\S+ ?)+)')
# FIELDS are of the form: 'xxxx = float'
FIELDS_RE = re.compile(r'\w+\s+=\s+\-?\d+\.?\d*[eE]?[\+\-]?\d*')
# DATES are of the form: 'nn/nn/nn'
DATE_RE = re.compile(r'\d+/\d+/\d+')
# DATA fields within a segment begin with '% LAT LON'
DATA_FIELDS_RE = re.compile(r'%\s+LAT\s+LON')

# Maps between what's given in the srcmod file, and the output fields
TAG_MAP = [
    ('EVENTTAG', 'tag'),
    ('EVENT', 'description'),
]

# There are a number of data fields from the header of a Srcmod file that are
# directly copied over into the output of the file reader. This is an array of
# the tuples where:
#
FIELD_MAP = [
    ('LAT', 'epicenterLatitude'),
    ('LON', 'epicenterLongitude'),
    ('DEP', 'depth'),
    ('MW', 'magnitude'),
    ('MO', 'moment'),
]

# Constants to do some conversions.
KM2M = 1e3  # Convert kilometers to meters
CM2M = 1e-2  # Convert centimeters to meters

def mean_angle(deg, w):
    #get mean angle, accounting for wraparound problem, based on https://rosettacode.org/wiki/Averages/Mean_angle#Python
    sumangles = 0.
    for i in range(len(deg)):
        sumangles += w[i]*rect(1, math.radians(deg[i]))
    average_angle = math.degrees(phase(sumangles/len(deg)))
    if average_angle<0: average_angle += 360.
    if average_angle>360: average_angle -= 360.
    return average_angle


def unit_normal(a, b, c):
    #unit normal vector of plane defined by points a, b, and c
    x = np.linalg.det([[1,a[1],a[2]],
                       [1,b[1],b[2]],
                       [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
                                          [b[0],1,b[2]],
                                          [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
                                          [b[0],b[1],1],
                                          [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)


def poly_area(poly):
    #area of polygon poly, from https://stackoverflow.com/questions/12642256/python-find-area-of-polygon-from-xyz-coordinates
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def _FindFields(data, opt_ignore_duplicate=True):
  """Finds all 'FIELD = VAL' in given string.

  Inputs:
    data: String of data to search for.
    opt_ignore_duplicate: We have two options if we encounter a named field more
      than once: we can ignore the duplicate, or we can take the new value. By
      default, we will ignore the duplicate fields.
  Returns:
    Dictionaries 'field': 'val' where 'val' has been cast to float. NB: unless
    specified, only the first field found is specified.
  """
  # Extract the fields from the data.
  fields = {}
  for field in FIELDS_RE.findall(data):
    name, val = field.split('=')
    name = name.strip().upper()
    # Take the first values seen.
    if not opt_ignore_duplicate or name not in fields:
      fields[name] = float(val.strip())
  return fields


def _SeparateSegments(num_segments, fields, data):
  """Pulls the segments out of the data.

  Depending on if the srcmod file is a multi or single segment file, this
  function will find the segment separator, and return the separated segment
  data.

  A single segment file looks like:

    % SRCMOD HEADER
    % SOURCE MODEL PARAMETERS
    %     [ SEGMENT_HEADER ]
    data

  A multi-segment file will look like:

    % SRCMOD HEADER
    % SEGMENT
    %     [ SEGMENT_HEADER ]
    data

    [.... num_segments ....]

    % SEGMENT
    %     [ SEGMENT_HEADER ]
    data

  Args:
    num_segments: The number of segments in the data.
    fields: The header of the srcmod file.
    data: The data (as a string) of the srcmod file.

  Returns:
    Tuple of (segments, segment_fields)
      segments: Array of all the segment data (as strings).
      segment_fields: The fields that have been stripped from the segment
        headers.
  """
  # Set up the segment data.
  if num_segments > 1:
    delimeter = '% SEGMENT'
    assert delimeter in data
    segments = [delimeter + _ for _ in data.split(delimeter)[1:]]
    segment_fields = [_FindFields(seg) for seg in segments]
  else:
    delimeter = '% SOURCE MODEL PARAMETERS'
    assert delimeter in data
    segments = [delimeter + _ for _ in data.split(delimeter)[1:]]
    segment_fields = [fields]

  assert len(segments) == num_segments
  assert len(segment_fields) == num_segments
  return segments, segment_fields


def _GetSegmentData(data):
  """Given a segment of data, we parse it into the appropriate fields.

  Args:
    data: String that contains all the characters in a segment's worth of data.
  Returns:
    List of lists of dictionaries.
  """
  ret = []
  rows = []
  names = []
  last_z = None
  for line in data.split('\n'):
    if not line: continue  # Skip blank lines
    if DATA_FIELDS_RE.match(line):  # Find field names
      # We extract the names of the fields.
      # The field names will be a in a string of the following form:
      #
      #     '%     F1   F2    F3==X     Z'
      #
      # First we split up the string by removing all spaces, discard the first
      # one ('%'), and then we remove any pieces after and including '=' in the
      # field name. NB: The last row must be a 'Z'
      names = [x.upper() for x in line.split()[1:]]
      names = [x.split('=')[0] if '=' in x else x for x in names]
    if line[0] == '%':  # Skip comment lines.
      continue
    else:
      # Make a dict of our values.
      val = {n: float(v) for n, v in zip(names, line.split())}
      assert -180. <= val['LON'] <= 180.
      assert -90. <= val['LAT'] <= 90.

      # If the z value we've just read in doesn't equal the last z value we've
      # read in, we have a new row. We then save off the row we've read so far
      # before adding the new value to the rows.
      if last_z is not None and val['Z'] != last_z:
        ret.append(rows)
        assert len(ret[0]) == len(ret[-1])  # Is same length as previous?
        rows = []
      rows.append(val)
      last_z = val['Z']
  if rows:
    ret.append(rows)
  assert len(ret[0]) == len(ret[-1])  # Is same length as previous?
  return ret


def ReadSrcmodFile(filename, FSBFilesFolder):
  """Reads a Srcmod file.
  Inputs: filename: Full path to Srcmod file.
  Returns: List of dictionaries. Each dictionary is a single segment of the fault.
  """
  print 'Reading SRCMOD file: ' + filename
  src_mod = collections.defaultdict(list)
  with open(filename, 'r') as f:
    data = f.read()
    # Read the date.
    date = DATE_RE.search(data).group(0)
    src_mod['date'] = date
    src_mod['datetime'] = datetime.datetime.strptime(date, '%m/%d/%Y')
    src_mod['areaTotal'] = 0.
    # Extract tags
    tags = {}
    for tag in TAGS_RE.findall(data):
      name, val = tag.split(':')
      tags[name.strip().upper()] = val.strip()

    # Remap tags to src_mod output.
    for in_name, out_name in TAG_MAP:
      if in_name not in tags:
        print 'error', in_name, tags
        continue
      src_mod[out_name] = tags[in_name]

    # Find fields, and remap them to src_mod output.
    fields = _FindFields(data)
    for in_name, out_name in FIELD_MAP:
      if in_name not in fields:
        print 'error', in_name, fields
        continue
      src_mod[out_name] = fields[in_name]

    # Calculate some epicenter projection stuff.
    _, _, number, letter = utm.from_latlon(src_mod['epicenterLatitude'],
                                           src_mod['epicenterLongitude'])
    src_mod['zoneNumber'] = number
    src_mod['zoneLetter'] = letter
    proj = pyproj.Proj(proj='utm', zone='{}{}'.format(number, letter),
                       ellps='WGS84')
    src_mod['projEpicenter'] = proj
    src_mod['epicenterXUtm'], src_mod['epicenterYUtm'] = proj(
        src_mod['epicenterLongitude'], src_mod['epicenterLatitude'])

    # Set up the segment data.
    num_segments = int(fields['NSG'])
    segments, segment_fields = _SeparateSegments(num_segments, fields, data)

    # Loop through the segments.
    for i in range(num_segments):
      if segment_fields[i].has_key('STRIKE'):
        seg_strike = segment_fields[i]['STRIKE']
      else:
        seg_strike = fields['STRK']
      angle = -(seg_strike-90)
      if angle < 0:
        angle += 360
            
      if segment_fields[i].has_key('DZ'): width = segment_fields[i]['DZ']
      elif fields.has_key('DZ'): width = fields['DZ']
      else:
          print 'no segment DZ given'
          assert False
          check.set_trace()
      if segment_fields[i].has_key('DX'): length = segment_fields[i]['DX']
      elif fields.has_key('DX'): length = fields['DX']
      else:
          print 'no segment Dx given'
          assert False
      data = _GetSegmentData(segments[i])

      # Calculate the geometric coordinates of the segments.
      #
      # In the following code, we convert the srcmod data into a format we use
      # for our coloumb stress calculations. Specifically, we take the srcmod
      # data and remap the geometry into a form we need. The original srcmod
      # data looks like:
      #
      #               v this coordinate is the x,y,z data point.
      #       +-------*--------+
      #       |                |
      #       |                |
      #       +----------------+
      #
      # The original srcmod data is also along a x,y,z coordinate system where
      # the Z vector is projected from the core of the earth. We need to
      # decompse the data (using the strikeslip and dipslip[*]) of the fault.
      #
      # The first thing we do is find the offsets between the x/y coordinates --
      # specifically, [xy]_top_offset and [xyz]_top_bottom_offset. We calculate
      # these values as follows:
      #
      #   [xy]_top_offset is calculated by assuming the fault patches are
      #     uniformally spaced, and sized on a given segment. Given this, and
      #     the length and angle of the fault, we calculate the offsets as the
      #     length rotated about the angle.
      #   [xyz]_top_bottom_offsets are calculated by (again assuming uniform
      #     patch size) taking the difference between two [xyz] coordinates.
      #
      # We remap the coordinates into the following format:
      #
      #       <---------------->  x_top_offset * 2
      #       |                |
      #
      # xyz1  +----------------+ xyz2  --^
      #       |                |         |  x_top_bottom_offset
      #       |                |         |
      # xyz3  +----------------+ xyz4  --v
      #
      # We do this remaping with a number of different transforms for x, y, and
      # z.
      #
      # [*] strikeslip is the angle the fault, and slip as the two plates move
      # laterally across each other. dipslip is the angle of the fault as the
      # two plates move under/over each other.
      
      rot = np.array([[math.cos(math.radians(angle)),
                       -math.sin(math.radians(angle))],
                      [math.sin(math.radians(angle)),
                       math.cos(math.radians(angle))]])
      x_orig = np.array([[length / 2.0], [0.0]])
      x_rot = np.dot(rot, x_orig)
      x_top_offset = x_rot[0] * KM2M
      y_top_offset = x_rot[1] * KM2M

      if len(data)>1:
        x_top_bottom_offset = (data[1][0]['X'] - data[0][0]['X']) * KM2M
        y_top_bottom_offset = (data[1][0]['Y'] - data[0][0]['Y']) * KM2M
        z_top_bottom_offset = (data[1][0]['Z'] - data[0][0]['Z']) * KM2M
        z_top_bottom_offset2 = np.abs(width*np.sin(math.radians(np.double(segment_fields[i]['DIP'])))) #use these to check method below, which we have to use when the segment only has one depth associated with the patches
        xo = np.abs(width*np.cos(math.radians(np.double(segment_fields[i]['DIP']))))
        R = np.array([[math.cos(math.radians(-1.0*seg_strike)), -math.sin(math.radians(-1.0*seg_strike))], [math.sin(math.radians(-1.0*seg_strike)), math.cos(math.radians(-1.0*seg_strike))]])
        [x_top_bottom_offset2, y_top_bottom_offset2] = np.dot(R, [xo, 0.])
        x_top_bottom_offset2 = x_top_bottom_offset2*KM2M
        y_top_bottom_offset2 = y_top_bottom_offset2*KM2M
        z_top_bottom_offset2 = z_top_bottom_offset2*KM2M
        assert np.abs(x_top_bottom_offset2-x_top_bottom_offset)<100.0 #are we within 100 meters? seems reasonable for rounding error
        assert np.abs(y_top_bottom_offset2-y_top_bottom_offset)<100.0
        assert np.abs(z_top_bottom_offset2-z_top_bottom_offset)<100.0
      else:
        z_top_bottom_offset = np.abs(width*np.sin(math.radians(np.double(segment_fields[i]['DIP'])))) #use these to check method below, which we have to use when the segment only has one depth associated with the patches
        xo = np.abs(width*np.cos(math.radians(np.double(segment_fields[i]['DIP']))))
        R = np.array([[math.cos(math.radians(-1.0*seg_strike)), -math.sin(math.radians(-1.0*seg_strike))], [math.sin(math.radians(-1.0*seg_strike)), math.cos(math.radians(-1.0*seg_strike))]])
        [x_top_bottom_offset, y_top_bottom_offset] = np.dot(R, [xo, 0.])
        x_top_bottom_offset = x_top_bottom_offset*KM2M
        y_top_bottom_offset = y_top_bottom_offset*KM2M
        z_top_bottom_offset = z_top_bottom_offset*KM2M

      # Loops over the down-dip and along-strike patches of the current panel
      for dip in range(0, len(data)):
        for strike in range(0, len(data[0])):
          # Extract top center coordinates of current patch
          x_top_center = data[dip][strike]['X'] * KM2M
          y_top_center = data[dip][strike]['Y'] * KM2M
          z_top_center = data[dip][strike]['Z'] * KM2M
          src_mod['patchLongitude'].append(data[dip][strike]['LON'])
          src_mod['patchLatitude'].append(data[dip][strike]['LAT'])

          # Calculate location of top corners and convert from km to m
          src_mod['x1'].append(x_top_center + x_top_offset)
          src_mod['y1'].append(y_top_center + y_top_offset)
          src_mod['z1'].append(z_top_center)
          src_mod['x2'].append(x_top_center - x_top_offset)
          src_mod['y2'].append(y_top_center - y_top_offset)
          src_mod['z2'].append(z_top_center)

          # Calculate location of bottom corners and convert from km to m
          src_mod['x3'].append(x_top_center + x_top_bottom_offset +
                               x_top_offset)
          src_mod['y3'].append(y_top_center + y_top_bottom_offset +
                               y_top_offset)
          src_mod['z3'].append(z_top_center + z_top_bottom_offset)
          src_mod['x4'].append(x_top_center + x_top_bottom_offset -
                               x_top_offset)
          src_mod['y4'].append(y_top_center + y_top_bottom_offset -
                               y_top_offset)
          src_mod['z4'].append(z_top_center + z_top_bottom_offset)

          # Create UTM version of the same
          x_top_center_utm, y_top_center_utm = proj(
              src_mod['patchLongitude'][-1], src_mod['patchLatitude'][-1])
          src_mod['patchXUtm'] = x_top_center_utm
          src_mod['patchYUtm'] = y_top_center_utm
          src_mod['x1Utm'].append(x_top_center_utm + x_top_offset)
          src_mod['y1Utm'].append(y_top_center_utm + y_top_offset)
          src_mod['z1Utm'].append(z_top_center)
          src_mod['x2Utm'].append(x_top_center_utm - x_top_offset)
          src_mod['y2Utm'].append(y_top_center_utm - y_top_offset)
          src_mod['z2Utm'].append(z_top_center)
          src_mod['x3Utm'].append(x_top_center_utm + (x_top_bottom_offset +
                                                      x_top_offset))
          src_mod['y3Utm'].append(y_top_center_utm + (y_top_bottom_offset +
                                                      y_top_offset))
          src_mod['z3Utm'].append(z_top_center + z_top_bottom_offset)
          src_mod['x4Utm'].append(x_top_center_utm + (x_top_bottom_offset -
                                                      x_top_offset))
          src_mod['y4Utm'].append(y_top_center_utm + (y_top_bottom_offset -
                                                      y_top_offset))
          src_mod['z4Utm'].append(z_top_center + z_top_bottom_offset)
          
          # Extract patch dip, strike, width, and length
          src_mod['dip'].append(segment_fields[i]['DIP'])
          src_mod['strike'].append(seg_strike)
          src_mod['rake'].append(data[dip][strike].get('RAKE', 'NaN'))
          src_mod['angle'].append(angle)
          src_mod['width'].append(KM2M * width)
          src_mod['length'].append(KM2M * length)
          src_mod['slip'].append(data[dip][strike]['SLIP'])
          # deal with wraparound problem for rakes and strikes that skews some header rakes in SRCMOD files, and deal with  problem of rakes of patches that do not slip but have fixed rakes of 45 degrees, for example, skewing the mean rake of the slip distribution
          src_mod['areaTotal'] = src_mod['areaTotal'] + KM2M*length*KM2M*width
          
          #verify length and width of patch are defined correctly with check of patch area
          v1 = [src_mod['x1Utm'][-1].tolist()[0], src_mod['y1Utm'][-1].tolist()[0], src_mod['z1Utm'][-1]]
          v2 = [src_mod['x2Utm'][-1].tolist()[0], src_mod['y2Utm'][-1].tolist()[0], src_mod['z2Utm'][-1]]
          v3 = [src_mod['x3Utm'][-1].tolist()[0], src_mod['y3Utm'][-1].tolist()[0], src_mod['z3Utm'][-1]]
          v4 = [src_mod['x4Utm'][-1].tolist()[0], src_mod['y4Utm'][-1].tolist()[0], src_mod['z4Utm'][-1]]
          if np.abs(KM2M*length*KM2M*width-poly_area([v1, v2, v4, v3]))>0.05*poly_area([v1, v2, v4, v3]): # check these areas are within 100000 sq meters of each other
              print 'patch area, defined by width and length, is not within 5% of size of actual patch size.'
              print 'width*length = ' + str(KM2M*length*KM2M*width) + ' square meters'
              print 'area of true patch = ' + str(poly_area([v1, v2, v4, v3])) + '. This is a difference of: ' + str(np.abs(KM2M*length*KM2M*width-poly_area([v1, v2, v4, v3]))) + '.'
              assert False

    src_mod['headerstrike'] = fields['STRK']

    # get weights for averaging rake, dip, and strike by amount of slip
    for i in range(len(src_mod['width'])):
        src_mod['weights'].append((np.double(src_mod['width'][i])*np.double(src_mod['length'][i]))/np.double(src_mod['areaTotal']))
    assert 0.99 < np.sum(np.double(src_mod['weights'])) < 1.01
    
    # deal with issue of rakes of patches that have zero slip with fixed rakes
    zero_slip_indexes = np.where(np.double(src_mod['slip'])==0)
    dipvec = np.delete(np.double(src_mod['dip']), zero_slip_indexes)
    strikevec = np.delete(np.double(src_mod['strike']), zero_slip_indexes)
    weightvec = np.delete(np.double(src_mod['weights']), zero_slip_indexes)
    src_mod['dipMean'] = mean_angle(dipvec, weightvec)
    src_mod['strikeMean'] = mean_angle(strikevec, weightvec)

    #deal with a few special cases
    if (filename == FSBFilesFolder + 's1995KOBEJA01HORI.fsp') or (filename == FSBFilesFolder + 's1995KOBEJA01SEKI.fsp') or (filename == FSBFilesFolder + 's1995KOBEJA01KOKE.fsp') or (filename == FSBFilesFolder + 's1995KOBEJA01WALD.fsp') or (filename == FSBFilesFolder + 's1995KOBEJA01YOSH.fsp') or (filename == FSBFilesFolder + 's1995KOBEJA02SEKI.fsp') or (filename == FSBFilesFolder + 's2010ELMAYO01WEIx.fsp'):
        src_mod['strikeMean'] = fields['STRK'] # for the few cases where two segments switch strike by 180 degrees, because one is dipping slightly one way and the other is dipping the other way
    if (filename == FSBFilesFolder + 's2010HAITIx01HAYE.fsp'): #same situation, strikes switching by 180 degrees; authors define header strike by taking the average strike. This leads to slip vectors perpendicular to all the faults. However, for this slip distribution and the 2010 Darfield distribution, mean strike/dip/rake are basically meaningless because the geometry is so complicated
        tmp = copy.copy(strikevec)
        tmp[np.where(tmp==257.)] = tmp[np.where(tmp==257.)]-180.
        src_mod['strikeMean'] = mean_angle(tmp, weightvec)
    
    src_mod['headerrake'] = fields['RAKE']
    if src_mod['headerrake'] > 360.: src_mod['headerrake'] = src_mod['headerrake']-360
    if src_mod['headerrake'] < 0.: src_mod['headerrake'] = src_mod['headerrake']+360

    #Substitute header rake when rake is not specified for each patch
    p = 0
    for item in src_mod['rake']:
        if 'NaN' == item:
            p += 1
    if p == len(src_mod['rake']):
        #Substituting header rake (' + str(src_mod['headerrake']) + ' degrees) for all patch rakes if author of slip distribution did not add them to each patch
        src_mod['rake'] = np.ones(np.shape(src_mod['rake']))*src_mod['headerrake']

    #process rakes as we did for strikes and dips above to get mean rake
    rakevec = np.delete(np.double(src_mod['rake']), zero_slip_indexes)
    src_mod['rakeMean'] = mean_angle(rakevec, weightvec)

    # deal with a special case
    if (filename == FSBFilesFolder + 's1999CHICHI01WUxx.fsp'):
        src_mod['rakeMean'] = src_mod['headerrake'] # in this slip distribution, there are a lot of patches slipping a little bit in the opposite direction of the main slip vector, so the mean rake for the Chichi distribution is 101 degrees, when everyone reports rakes more like 55 degrees, so we use the header rake

    # check that negative rakes are not messing things up
    for rake in src_mod['rake']:
        if -.01 <= rake <= 360.01: continue
        else:
            x = copy.copy(np.double(src_mod['rake']))
            x[x>360.] =x[x>360.]-360.
            x[x<0.] =x[x<0.]+360.
            xf = np.delete(x, zero_slip_indexes)
            assert src_mod['rakeMean']-2. <= mean_angle(xf, src_mod['weights']) <= src_mod['rakeMean']+2.

    #Calculate slip in strike and dip direction, assuming rake is defined counterclockwise from strike: 90 degree rake is thrust fault, -90 is normal fault, 0 or 360 is a left-lateral strike slip fault, -180 or 180 is a right-lateral strike slip fault. So, positive strike slip is left-lateral and positive dip slip is thrust sense motion.
    c = 0
    for i in range(num_segments):
       data = _GetSegmentData(segments[i])
       for dip in range(0, len(data)):
          for strike in range(0, len(data[0])):
            # Extract fault slip
            rot = np.array([[math.cos(math.radians(src_mod['rake'][c])),
                           -math.sin(math.radians(src_mod['rake'][c]))],
                          [math.sin(math.radians(src_mod['rake'][c])),
                           math.cos(math.radians(src_mod['rake'][c]))]])
            x_orig = np.array([[src_mod['slip'][c]], [0]])
            x_rot = np.dot(rot, x_orig)
            src_mod['slipStrike'].append(x_rot[0])
            src_mod['slipDip'].append(x_rot[1])
            c += 1

  # Check that our dips and strikes are within proper ranges.
  for dip in src_mod['dip']:
    assert 0. <= dip <= 90.
  for strike in src_mod['strike']:
    assert 0. <= strike <= 360.

  print 'Done reading SRCMOD file ' + filename

  return src_mod
