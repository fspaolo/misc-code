#!/usr/bin/env python

# region.py - Extracts data in a specified region: 
# west, east, north, south.
# 
# Author: Fernando Paolo 
# Date: Jun/2008.
# Usage: python region.py -h

import numpy as N
import optparse
from sys import argv, exit
from os import path
from scipy import weave
from scipy.weave.converters import blitz

usage = "python %prog <filein> [options]"
epilog = "Input file: lon [0/360] or [-180/+180], lat [-90/+90]."
parser = optparse.OptionParser(usage=usage, epilog=epilog)

parser.add_option('-R', 
    	          dest='region',
                  default='(308,330,-6,8)', 
                  help='region to extract pts [deg]: -Rwest,east,south,north',
                  )
parser.add_option('-r', 
    	          dest='inout',
                  default=1, 
				  type='int',
                  help='select pts inside (1) or outside (0) the region: -r1',
                  )
parser.add_option('-o', 
    	          dest='fileout',
                  default='region.out', 
                  help='write output to FILEOUT: -oregion.out',
                  )
parser.add_option('-x', 
    	          dest='colx',
                  default=0, 
				  type='int',
                  help='longitude column in the file: -x0',
                  )
parser.add_option('-y', 
    	          dest='coly',
                  default=1, 
				  type='int',
                  help='latitude column in the file: -y1',
                  )
parser.add_option('-c',
    	          dest='change',
                  action='store_true',
                  default=False, 
                  help='change input [lat,lon] to [lon,lat] (or vice-versa): -c',
                  )
options, remainder = parser.parse_args()

if len(remainder) < 1:
    print 'The input file is missing!\n'
    parser.print_help()
    exit()

filein = remainder[0]
fileout = options.fileout
region = eval(options.region)
inout = options.inout
colx = options.colx
coly = options.coly
change = options.change

west, east, south, north = region

# converts longitude
if west < 0:
    west += 360
if east < 0:
    east += 360

#--------------------------------------------------------------------------

def select_region(IN, OUT, west, east, south, north, colx, coly, inout):

    """Selects the pts inside the specified region"""

    nrow = IN.shape[0]
    ncol = IN.shape[1]
    code = \
    """
	int k = 0;
	double x, y;
    for (int i = 0; i < nrow; i++) {
	    x = IN(i,colx);
	    y = IN(i,coly);

		// selects points inside the region
        if (inout == 1) {
	        if ((south < y) && (y < north) && (west < x) && (x < east)) {
	            for (int j = 0; j < ncol; j++)
	                OUT(k,j) = IN(i,j);
	            k++;
	        }
        }
		// selects points outside the region
		else {
	        if ((y < south) || (north < y) || (x < west) || (east < x)) {
	            for (int j = 0; j < ncol; j++)
	                OUT(k,j) = IN(i,j);
	            k++;
	        }
		}
	}

	return_val = k;  // number of selected pts
	"""
    vars = ['IN', 'OUT', 'west', 'east', 'south', 'north', 'colx', \
	        'coly', 'nrow', 'ncol', 'inout']
    return weave.inline(code, vars, type_converters=blitz)


#--------------------------------------------------------------------

def lon_360to180(MAT, colx):

    """Converts LON from 0/360 to -180/180 (only if LON > 180)"""

    nrow = MAT.shape[0]
    code = \
    """
	double lon;
	for (int i = 0; i < nrow; i++) {
        lon = MAT(i,colx);
	    if (lon > 180)
	        MAT(i,colx) = lon - 360.;
	}
	"""
    weave.inline(code, ['MAT', 'colx', 'nrow'], type_converters=blitz)

#--------------------------------------------------------------------------

def lon_180to360(MAT, colx):

    """Converts LON from -180/180 to 0/360 (only if LON < 0)"""

    nrow = MAT.shape[0]
    code = \
    """
	double lon;
	for (int i = 0; i < nrow; i++) {
        lon = MAT(i,colx);
	    if (lon < 0)
	        MAT(i,colx) = lon + 360.;
	}
	"""
    weave.inline(code, ['MAT', 'colx', 'nrow'], type_converters=blitz)

#--------------------------------------------------------------------------

def main():

    print 'Loading data ...'
    IN = N.loadtxt(filein)
    lon_180to360(IN, colx)       # converts longitude
    rows = IN.shape[0]
    cols = IN.shape[1]
    AUX = N.empty((rows,cols), 'float64')

    if inout == 1:
        print 'selecting pts inside the region ...'
    else:
        print 'selecting pts outside the region ...'

    pts = select_region(IN, AUX, west, east, south, north, colx, coly, inout)
    PTS = AUX[:pts,:]

    # change columns: lat,lon -> lon,lat
    if change == True:
        TMP = PTS[:,0].copy()   # independent copy of lat
        PTS[:,0] = PTS[:,1]     # lon
        PTS[:,1] = TMP[:]
        print 'columns of input file changed'

    N.savetxt(fileout, PTS, fmt='%f', delimiter=' ')
    print 'npts: %d' % PTS.shape[0]
    print 'output -> ' + fileout


if __name__ == '__main__':
    main()
