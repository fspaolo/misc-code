"""
 Extract header information from IDR binary files. 
 
 Options:
 * print the header (by default)
 * print the name of files within a given time window
 * print the name of files within a given year
 * print the time span by given files (it can be the whole data base)

 For IDR format see:
 http://icesat4.gsfc.nasa.gov/data_products/level2.html

 Fernando Paolo <fpaolo@ucsd.edu>
 December, 2009
"""

import numpy as np
import argparse as ap
import os
import sys

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='IDR binary file[s] to read')
parser.add_argument('-t', dest='twindow', default=None, 
		    help='time window (YYYYMM/YYYYMM) to select files [all]')  
parser.add_argument('-y', dest='year', type=int, default=None, 
		    help='year (YYYY) to select files [all]')  
parser.add_argument('-s', dest='tspan', default=None, action='store_const',
                    const=True, help='print the time span by files')

args = parser.parse_args()
files = args.file
twindow = args.twindow
year = args.year
tspan = args.tspan

if twindow is not None:
    yyyymm0, yyyymm1 = [int(t) for t in twindow.split('/')]

outfile = 'idrheader.info'

#----------------------------------------------------------------

IDRH = [                 # 100-byte IDR Header
    ('id', '|S2'),       # 0
    ('revdir', '|S14'),  # 1
    ('geodir', '|S14'),  # 2
    ('bindir', '|S14'),  # 3
    ('version', '>i4'),  # 4
    ('datebeg', '>i4'),  # 5
    ('timebeg', '>i4'),  # 6
    ('dateend', '>i4'),  # 7
    ('timeend', '>i4'),  # 8
    ('satid', '>i4'),    # 9
    ('coverage', '|S8'), # 10
    ('spares', '|S24'),  # 11
]

idr = np.dtype(IDRH)

def get_yyyy(date): 
    y = int(date/10000.)
    if y <= 99:          # YY format is 20th century
        y = y + 1900
    return y

def get_yyyymm(date): 
    ym = int(date/100.)
    if ym <= 9912:       # YYMM format is 20th century
        ym = ym + 190000
    return ym

def main():
    print 'reading files: %d... ' % len(files)

    if twindow is not None or year is not None:
        f = open(outfile, 'w')

    # iterate over input files
    begmin = np.inf
    endmax = 0 
    nfiles = 0
    for ifname in files:
        # load the whole data file in-memory <-- careful with this!
        data = np.fromfile(ifname, dtype=idr, count=-1)

        # iterate over records (100B IDR structs)
        nrec = len(data)
        for i in xrange(nrec):
            id = data[i][0]

            # IDR Header Record
            if id == 'IH':

                if twindow is not None:             # look for time window
                    datebeg = data[i][5]            # beg of file
                    dateend = data[i][7]            # end of file
                    yyyymmbeg = get_yyyymm(datebeg) # check beg of file only!
                    if yyyymm0 <= yyyymmbeg and yyyymmbeg <= yyyymm1: 
                        f.write('%s\n' % ifname)
			f.write('date [beg end]: %d %d\n' % (datebeg, dateend))
                        print ifname,
                        sys.stdout.flush()
                    break

                elif year is not None:              # look for same years
                    datebeg = data[i][5]
                    dateend = data[i][7]
                    yyyybeg = get_yyyy(datebeg)     # check beg of file only!
                    if year == yyyybeg:                
                        f.write('%s\n' % ifname)
			f.write('date [beg end]: %d %d\n' % (datebeg, dateend))
                        print ifname,
                        sys.stdout.flush()
                    break

                elif tspan is not None:             # look for time tspan of db
                    datebeg = data[i][5]
                    dateend = data[i][7]
                    if datebeg < begmin:
                        begmin = datebeg            # first file
                        endmin = dateend
                    elif dateend > endmax:
                        begmax = datebeg            # last file
                        endmax = dateend
                    break
                    
                else:
                    #--------------------------------------------
                    print '\nHEADER RECORD'
                    print 'rev accsess dir:', data[i][1]
                    print 'georeference dir:', data[i][2]
                    print 'bin/rev dir:', data[i][3]
                    print 'version:', data[i][4]
                    print 'date beg of file YYMMDD:', data[i][5]
                    print 'time beg of file HHMMSS:', data[i][6]
                    print 'date end of file YYMMDD:', data[i][7]
                    print 'time end of file HHMMSS:', data[i][8]
                    print 'satellite ID:', data[i][9]
                    print 'coverage by db:', data[i][10]
                    #---------------------------------------------
                    break  # to read only one header per file
            
    print '\ndone!'
    if twindow is not None or year is not None:
        f.close()
        print 'output: %s' % outfile 
    if tspan is not None:
        print 'time coverage (beg end):'
        print 'first file:', begmin, endmin
        print 'last file: ', begmax, endmax


if __name__ == '__main__':
    main()
