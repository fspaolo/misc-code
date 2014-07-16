"""
 Read IDR binary data files, calculate the year and gives the
 name of the corresponding files.

 Fernando Paolo <fpaolo@ucsd.edu>
 December, 2009
"""

import numpy as np
import argparse as ap
import sys

import idrs

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='IDR binary file[s] to read')
parser.add_argument('yyyy', dest='yyyy', type=int 
		    help='year (YYYY) to select files [all]')  
args = parser.parse_args()
files = args.file
yyyy = args.yyyy

idr = np.dtype(idrs.IDRD)

#----------------------------------------------------------------

MJD85 = 46066. # 1-Jan-1985 in MJD

def utc2year(utc85):
    yearsec = 31556926.             # 1 year in secs
    year85 = utc85 / yearsec        # years from 1985 (float)
    year = np.floor(year85 + 1985)  # present year
    return year 

def main():

    print 'reading files: %d... ' % len(files)

    # iterate over input files
    nfiles = 0
    nptsread = 0
    nptsvalid = 0
    for ifname in files:
        # load the whole data file in-memory <-- careful with this!
        data = np.fromfile(ifname, dtype=idr, count=-1)

        # iterate over records (100B IDR structs)
        npts = 0
        datebeg = np.inf
        dateend = 0
        nrec = len(data)
        for i in xrange(nrec):
            id = data['id'][i]

            # IDR Rev Record
            if id == 'IR':
                #-------------------------------------------------------
                mjd = data['lat'][i]             # days (integer part)
                secrev = data['lon'][i]          # secs (integer part) 
                fsecrev = data['surf'][i] / 1e6  # secs (fractional part)
                #-------------------------------------------------------

            # IDR Data Record
            elif id == 'ID': 
                #-------------------------------------------------------
                secdat = data['time'][i] / 1e6 # secs (since time in Rev)
                surfcheck = data['surf'][i]    # check whether the surf is valid
                #-------------------------------------------------------
                nptsread += 1

		if surfcheck != -9999:

                    # fday: fraction of a day
                    # mjd: modified julian days
                    # utc85: time in seconds since 1-Jan-1985 0h
                    fday = (secrev + fsecrev + secdat) / 86400.
                    utc85 = ((mjd - MJD85) + fday) * 86400.

                    year = utc2year(utc85)
                    if year == yyyy:
                        print year 
                    nfiles += 1

    print 'done!'
    print 'files listed:', nfiles
    print 'points read:', nptsread


if __name__ == '__main__':
    main()
