"""
 Read IDR binary data files:

 - select points in a given geographic region
 - unapply tide correction (if applied)
 - applied increment for orbit correction (if defined)
 - filter out undefined elevation values
 - save to HDF5 or ASCII format
 
 Notes:
 Seasat - use increment 2
 ERS-1/2 - use increment 3
 value = -9999 is undefined
 correction = 32767 is undefined

 for IDR format see:
 http://icesat4.gsfc.nasa.gov/data_products/level2.html

 Fernando Paolo <fpaolo@ucsd.edu>
 December, 2009
"""

import numpy as np
import tables as tb
import string as str
import argparse as ap
from bitarray import bitarray
from datetime import date
import os
import sys

import idrs

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='IDR binary file[s] to read')
parser.add_argument('-r', dest='region', default='-180/180/-90/90', 
                    help='select region: left/right/bottom/upper')
parser.add_argument('-txt', dest='ext', default='h5', action='store_const',
                    const='txt', help='for ASCII output [default HDF5]')
args = parser.parse_args()
files = args.file
region = args.region
ext = args.ext

idr = np.dtype(idrs.IDRD)

metadata = 'Fernando_Paolo <fpaolo@ucsd.edu> %s SIO/UCSD' % date.today()

#----------------------------------------------------------------

MJD1985 = 46066. # 1-Jan-1985 in MJD

def check_region(left, right, bottom, top):
    """Check if input region is correct."""
    if left >= right or left < -180 or right > 360:
        print 'error: region: left >= right or out of bounds'
        sys.exit()
    elif bottom >= top or bottom < -90 or top > 90:
        print 'error: region: bottom >= top or out of bounds'
        sys.exit()
    elif left < 0 and right >= 0:   # 0/360 -> +/-180
        islon360 = False 
        return left, right, bottom, top, islon360
    else:                           # +/-180 -> 0/360
        if left < 0: 
            left += 360.
        if right < 0: 
            right += 360.
        islon360 = True 
        return left, right, bottom, top, islon360


def main():

    # get region 
    left, right, bottom, top = [float(s) for s in region.split('/')]
    left, right, bottom, top, islon360 = check_region(left, right, bottom, top)

    print 'processing files: %d... ' % len(files)

    # iterate over input files
    nfiles = 0
    nptsRead = 0
    nptsValid = 0
    for ifname in files:
        # load the whole data file in-memory
        data = np.fromfile(ifname, dtype=idr, count=-1)

        # iterate over records (100B IDR structs)
        createFile = False
        ismatCreated = False 
        npts = 0
        nrec = len(data)
        for i in xrange(nrec):
            id = data['id'][i]

            # IDR Rev Record
            if id == 'IR':
                #-------------------------------------------------------
                orbit = data['time'][i]          # orbit number
                mjd = data['lat'][i]             # days (integer part)
                secRev = data['lon'][i]          # secs (integer part) 
                fsecRev = data['surf'][i] / 1e6  # secs (fractional part)
                #-------------------------------------------------------

            # IDR Data Record
            elif id == 'ID': 
                #-------------------------------------------------------
                secDat = data['time'][i] / 1e6 # secs (since time in Rev)
                lat = data['lat'][i] / 1e6     # latitude (deg)
                lon = data['lon'][i] / 1e6     # longitude (deg)
                surf = data['surf'][i] / 1e2   # surface elevation (m)
                otide = data['otide'][i] / 1e3 # ocean tide correction (m)
                inc = data['inc2'][i] / 1e2    # orbit correction (m)
                inccheck = data['inc2'][i]     # check whether the inc is valid
                surfcheck = data['surf'][i]    # check whether the surf is valid
                altstat = data['altstat'][i]   # altimetry status flags (31-0)
                surfstat = data['surfstat'][i] # surface status flags (31-0)
                retstat = data['retstat1'][i]  # retracking status flags (15-0)
                #-------------------------------------------------------
                nptsRead += 1

                ###
                #print
                #print lat
                #print lon
                #print surf
                #print otide 
                #print surfcheck
                #print 
                #n = 32768 + 1024 + 64
                #n = retstat 
	        #b = bin(n)[2:]
	        #b = b[::-1]
                #print n
                #print b
                #print len(b)
                #for i in range(16):
                #    print i, b[i]
                #sys.exit()
                ###
                
                if islon360 and lon < 0: 
                    lon += 360
                elif lon > 180: 
                    lon -= 360

                ### computation

                # select pts
                if  surfcheck != -9999 and \
                    left <= lon and lon <= right and \
                    bottom <= lat and lat <= top:

                    createFile = True

                    # fday: fraction of a day
                    # mjd: modified julian days
                    # utc1985: time in seconds since 1-Jan-1985 0h
                    fday = (secRev + fsecRev + secDat) / 86400.
                    utc1985 = ((mjd - MJD1985) + fday) * 86400.

                    # detide and add increment
                    elev = surf
                    #if f24otide: 
                    #    elev = elev + otide YES -> use flag !!!!!!!!!!!!!!!!!
                    # do we want the data w/o inc? !!!!!!!!!!!!!!!!!!!!!!!!!!!
                    if inccheck != 32767: 
                        elev = elev + inc

                    if not islon360 and lon < 0: 
                        lon += 360    # 0/360

                    ### output
                    
                    dpoint = np.array([orbit, utc1985, lat, lon, elev])
                    npts += 1

                    if not ismatCreated:
                        ncol = 5                        # ncol = variables 
                        nrow = 1000                     # nrow = guess
                        odata = np.empty((nrow,ncol), 'f8')  
                        ismatCreated = True

                    if ismatCreated and npts <= nrow:           
                        odata[npts-1,:] = dpoint        # index starts at 0
                    elif ismatCreated and npts > nrow:  # more memory needed
                        tmp = np.empty((nrow,ncol), 'f8')
                        odata = np.vstack((odata, tmp))
                        odata[npts-1,:] = dpoint 
                        nrow = 2*nrow                   
                else:
                    continue
        
        if createFile and ext == 'h5':                  # HDF5 
            fname = str.join((ifname, '.', ext), '')
            h5f = tb.openFile(fname, 'w')    
            h5f.createArray(h5f.root, 'data', odata[:npts,:], metadata)            
            h5f.close()
            nfiles += 1
            nptsValid += npts 
        elif createFile and ext == 'txt':               # ASCII
            fname = str.join((ifname, '.', ext), '')
            np.savetxt(fname, odata[:npts,:], fmt='%f')
            nfiles += 1
            nptsValid += npts 
    
    print 'done!'
    print 'points read:', nptsRead
    print 'valid points:', nptsValid
    print 'files created:', nfiles
    print 'output extension: .%s' % ext


if __name__ == '__main__':
    main()
