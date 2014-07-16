"""
 Read IDR binary data files and separate them by year 
 (writting the names on an output file).

 Fernando Paolo <fpaolo@ucsd.edu>
 December, 2009
"""

import numpy as np
import argparse as ap
import sys

from _idryear import *

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='IDR binary file[s] to read')
parser.add_argument('-y', type=int, dest='yyyy', help='year to select files')
args = parser.parse_args()
files = args.files
yyyy = args.yyyy

idr = np.dtype(IDRD)

fout = 'idryear.out'

#----------------------------------------------------------------

def main():

    f = open(fout, 'w')

    print 'reading files: %d... ' % len(files)

    # iterate over input files
    nfiles = 0
    nptsread = 0
    nptsvalid = 0
    for ifname in files:
        # load the whole IDR file in-memory <-- careful with this!
        IDRs = np.fromfile(ifname, dtype=idr, count=-1)
        nrec = IDRs.shape[0]

        ID = np.empty(nrec, 'S2')        
        ID[:] = IDRs['id'][:]
        #--------------------------------------------------------
        MJD = np.empty(nrec, 'f8')
        MJD[:] = IDRs['lat'][:]             # days (integer part)
        SECREV = np.empty(nrec, 'f8')
        SECREV[:] = IDRs['lon'][:]          # secs (integer part) 
        FSECREV = np.empty(nrec, 'f8')
        FSECREV[:] = IDRs['surf'][:] / 1e6  # secs (fractional part)
        #--------------------------------------------------------
        SECDAT = np.empty(nrec, 'f8')
        SECDAT[:] = IDRs['time'][:] / 1e6   # secs (since time in Rev)
        SURFCHECK = np.empty(nrec, 'f8')
        SURFCHECK[:] = IDRs['surf'][:]      # check whether the point is valid
        #--------------------------------------------------------
        
        a = np.empty(nrec, 'S2')
        a[:] = ID[:]
        test(a)
        sys.exit()
        # iterate over records (IDR structs)
        res = iterate_over_recs(ID, MJD, SECREV, FSECREV, SECDAT, SURFCHECK)

        yearbeg, yearend, nr, nv = res
        nptsread += nr
        nptsvalid += nv

        if yearbeg == yearend:
            f.write('%s ' % ifname)
            nfiles += 1
        else:
            f.write('\n\n')
            f.write('%s ' % ifname)
            nfiles += 1

    f.write('\n\n')
    f.write('files listed: %d\n' % nfiles )
    f.write('points read: %d\n' % nptsread )
    f.write('points valid: %d\n' % nptsvalid )
    f.close()

    print 'done!'
    print 'files listed:', nfiles
    print 'points read:', nptsread
    print 'points valid:', nptsvalid
    print 'output:', fout


if __name__ == '__main__':
    main()
