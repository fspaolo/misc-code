#!/usr/bin/python

# Author: Fernando Paolo
# Date: Sep/2009

#================================================================
# ATENTION: the script [for some reason] only works with 
# Python 2.6 (Mac OSX conflict with pyhdf)
#
# usage: python2.6 <hdfFiles>
#================================================================

from pyhdf.SD import *
from pyhdf.HDF import ishdf
from datetime import date
import numpy as np
import sys
import os

#----------------------------------------------------------------
# definition of some variables
#----------------------------------------------------------------

# NOTE: you may run first the 'hdfstruct.py' script to figure
# out what the HDF structure looks like, i.e., the name and
# position of the variables in the file

# output directory path
OUTDIR = '/Users/fpaolo/sat_data/icesat/amery'

# region to be extracted: west/east/south/north
#REGION = '240.0/305.0/-76.0/-60.0'  # antp
REGION = '60.0/80.0/-82.0/-67.0'    # amery
#REGION = '110.0/120.0/-70.0/-65.7 ' # totten

# name used for longitude and latitude
LON_NAME = 'lon'
LAT_NAME = 'lat'

# sentence to be written in the header (output file)
HEADER = '' #ICESat, amery, Fernando Paolo, %s, SIO/UCSD' % date.today()

# symbol used for writing the header
COMMENT = '#'

#----------------------------------------------------------------

def usage():
    print '\nusage: python2.6 %s <hdfFiles>' % sys.argv[0]
	print 'obs: see the code for edition\n'
    sys.exit()


def lon_180to360(array):
    # convert lon from -180/+180 to 0/360 only if lon < 0
    for lon in array:
        if lon < 0:
            array[lon] += 360
    return array


def select_region(lons, lats, region):
    """Return the indices of lons/lats within the region"""    

    # region to be extracted
    west, east, south, north = region.split('/')
    west = float(west)
    east = float(east)
    south = float(south)
    north = float(north)
    # convert longitudes if needed (when lon < 0)
    if west < 0:
        west += 360
    if east < 0:
        east += 360
    lons = lon_180to360(lons)  
    # find the indices of values wihtin the specified region
    ind, = np.where( (west < lons) & (lons < east) & \
    		         (south < lats) & (lats < north) )
    return ind


def hdf2txt(hdfFile, txtFile):
    try:
        hdf = SD(hdfFile)        # open the HDF file
        attr = hdf.attributes()  # get global attribute dictionary
        dsets = hdf.datasets()   # get dataset dictionary
    except HDF4Error, msg:
        print "HDF4Error", msg

    if len(dsets) > 0:
        # sort dsets according their index (in the last position)
        dlist = sorted(dsets.items(), key=lambda(k,v):v[-1])
        dsNames = [k for k, v in dlist]

        # get all the longitudes and latitudes
        lons = hdf.select(LON_NAME).get()
        lats = hdf.select(LAT_NAME).get()
        ind = select_region(lons, lats, REGION)

        # if there is any point inside the region
        if len(ind) > 0:
            # open an ASCII file for writting
            txt = open(txtFile, 'w')
            
            # print the header
            if HEADER is not '':
                txt.write('%s %s\n' % (COMMENT, HEADER))
                if len(attr) > 0:
                    txt.write('%s %s\n' % (COMMENT, str(attr)))
                txt.write('%s %s\n' % (COMMENT, str(dsNames)))
            
            # get the datasets (selected pts) into a matrix
            data = hdf.select(dsNames[0]).get()[ind]  # get the first field
            if len(dsNames) > 1:
                for name in dsNames[1:]:
                    d = hdf.select(name).get()[ind]
                    data = np.column_stack((data, d))
            
            # save data to an ASCII file
            np.savetxt(txt, data, fmt='%f')
            txt.close()
            return 1  # if a file was created
    return 0


def main(args):
    # verify the arguments
    if len(args) < 2:
        usage()
    else:
        # process several HDF files
        infiles = args[1:]
        n_hdf = 0
        n_nohdf = 0
        print 'processing %d file(s) ...' % len(infiles)

        for hdfFile in infiles:
            if ishdf(hdfFile):
			    # split path -> head, tail
                head, tail = os.path.split(hdfFile)  
			    # split file -> root, ext
                root, ext = os.path.splitext(tail)
                # create the output file
                txtFile = os.path.join(OUTDIR, root + '.txt')
                # if a TXT file was created returns 1, otherwise 0                
                n_hdf += hdf2txt(hdfFile, txtFile)
            else:
                n_nohdf += 1

        if n_nohdf > 0:
            print '%d file(s) not in HDF format' % n_nohdf       
        if n_hdf > 0:
            print '%d HDF file(s) converted to ASCII' % n_hdf
            print 'output ->', os.path.dirname(txtFile) + '/same_input_name.txt'
        else:
            print 'no points inside the region'


if __name__ == '__main__':
    main(sys.argv)
