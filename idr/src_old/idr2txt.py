"""
 Read IDR binary data files:

 - select points in a given geographic region
 - unapply tide correction (if applied)
 - applied increment for orbit correction (if defined)
 - filter out undefined elevation values
 - show Header and Processing information (if selected)
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
from datetime import date
import os
import sys

import idrs

#----------------------------------------------------------------

getinfo = 'IH'

# convert to dtype 
if getinfo is None:
    idr = np.dtype(idrs.IDRD)
elif getinfo == 'IH':
    idr = np.dtype(idrs.IDRH)
elif getinfo == 'IP':
    idr = np.dtype(idrs.IDRP)

ext = 'h5'
M = 50000  # guess
N = 5

# region to be extracted: left/right/bottom/top
region = '-180/180/-90/90'  # antp
#region = '60.0/80.0/-82.0/-67.0'  # amery
#region = '110.0/120.0/-70.0/-65.7 ' # totten

# header
metadata = 'Fernando_Paolo <fpaolo@ucsd.edu> %s SIO/UCSD' % date.today()

#----------------------------------------------------------------

MJD1985 = 46066. # 1-Jan-1985 in MJD

def usage():
    print 'usage: python %s <ifnames>' % sys.argv[0]
    print '-> see the code for edition'
    sys.exit()

def check_region(left, right, bottom, top):
    """Check the input region."""
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


def main(args):
    # verify the arguments
    if len(args) < 2:
        usage()
    
    M = 50000

    # get files
    files = args[1:]

    # get region 
    left, right, bottom, top = [float(s) for s in region.split('/')]
    left, right, bottom, top, islon360 = check_region(left, right, bottom, top)

    sys.stdout.write('processing files: %d... ' % len(files))
    sys.stdout.flush()

    # iterate over input files
    nfiles = 0
    nptsRead = 0
    nptsValid = 0
    for ifname in files:
        # load the whole data file in-memory
        data = np.fromfile(ifname, dtype=idr, count=-1).byteswap()
        
        # iterate over records (100B IDR structs)
        createFile = False
        ismatCreated = False 
        npts = 0
        nrec = len(data)
        for i in xrange(nrec):
            id = data['id'][i]

            # IDR Header Record
            if getinfo == 'IH' and id == 'IH':
                #-------------------------------------------------------
                print '\nHEADER REC'
        	print 'rev accsess dir:', data['revDir'][i]   
        	print 'georeference dir:', data['geoDir'][i]   
        	print 'bin/rev dir:', data['binDir'][i]   
        	print 'version:', data['version'][i]   
        	print 'date beg of file YYMMDD:', data['dateBeg'][i]   
        	print 'time beg of file HHMMSS:', data['timeBeg'][i]   
        	print 'date end of file YYMMDD:', data['dateEnd'][i]   
        	print 'time end of file HHMMSS:', data['timeEnd'][i]   
        	print 'satellite ID:', data['satId'][i]   
        	print 'coverage by db:', data['coverage'][i]   
                #-------------------------------------------------------
                break

            # IDR Processing Record
            if getinfo == 'IP' and id == 'IP':
                #-------------------------------------------------------
                print '\nPROCESSING REC'
        	print 'date of proc YYMMDD:', data['dateProc'][i]   
        	print 'name/vers of proc prog:', data['procProg'][i]   
        	print 'input file #1:', data['infile1'][i]
        	print 'other input files:', data['infiles'][i]   
                #-------------------------------------------------------
                break
            
            # IDR Rev Record
            if getinfo is None and id == 'IR':
                #-------------------------------------------------------
                orbit = data['time'][i]          # orbit number
                mjd = data['lat'][i]             # days (integer part)
                secRev = data['lon'][i]          # secs (integer part) 
                fsecRev = data['surf'][i] / 1e6  # secs (fractional part)
                #-------------------------------------------------------

            # IDR Data Record
            if getinfo is None and id == 'ID': 
                #-------------------------------------------------------
                secDat = data['time'][i] / 1e6 # secs (since time in Rev)
                lat = data['lat'][i] / 1e6     # latitude (deg)
                lon = data['lon'][i] / 1e6     # longitude (deg)
                surf = data['surf'][i] / 1e2   # surface elevation (m)
                otide = data['otide'][i] / 1e3 # ocean tide correction (m)
                inc = data['inc2'][i] / 1e2    # orbit correction (m)
                incCheck = data['inc2'][i]     # check whether the inc is valid
                surfCheck = data['surf'][i]    # check whether the surf is valid
                surfStat = data['surfStat'][i] # surface status flags (31-0)
                retStat = data['retStat1'][i]  # retracking status flags (15-0)
                #-------------------------------------------------------
                nptsRead += 1
                
                if islon360 and lon < 0: 
                    lon += 360
                elif lon > 180: 
                    lon -= 360

                ### computation

                # select pts
                if  surfCheck != -9999 and \
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
                    #    elev = elev + otide YES -> use flag !!!!!!!!!!!!
                    # do we want the data w/o inc? !!!!!!!!!!!!!!!!!!!!!!!
                    if incCheck != 32767: 
                        elev = elev + inc

                    if not islon360 and lon < 0: 
                        lon += 360    # 0/360

                    ### output
                    
                    dpoint = np.array([orbit, utc1985, lat, lon, elev])
                    npts += 1

                    if not ismatCreated:
                        odata = np.empty((M,N), 'f8')
                        ismatCreated = True

                    if ismatCreated and npts <= M:           
                        odata[npts-1,:] = dpoint     # index starts at 0
                    elif ismatCreated and npts > M:  # more memory needed
                        tmp = np.empty((2*M,N), 'f8')
                        tmp[:npts-1,:] = odata        # try vstack !!!!!!!!!!
                        tmp[npts,:] = dpoint 
                        odata = tmp.copy()
                        M = 2*M
        
        if getinfo is not None:
            continue 

        if createFile and ext == 'h5':     # HDF5 
            fname = str.join((ifname, '.', ext), '')
            h5f = tb.openFile(fname, 'w')    
            h5f.createArray(h5f.root, 'data', odata[:npts,:])            
            h5f.close()
            nfiles += 1
            nptsValid += npts 
        elif createFile and ext == 'txt':  # ASCII
            fname = str.join((ifname, '.', ext), '')
            np.savetxt(fname, odata[:npts,:], 'f')
            nfiles += 1
            nptsValid += npts 
    
    print 'done!'
    print 'points read:', nptsRead
    print 'valid points:', nptsValid
    print 'files created:', nfiles
    print 'output extension: .%s' % ext


if __name__ == '__main__':
    main(sys.argv)
