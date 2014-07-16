"""
 Read IDR binary data files and separate them by year 
 (writting the names to an output file).

 Fernando Paolo <fpaolo@ucsd.edu>
 December, 2009
"""

import numpy as np
import argparse as ap
import math
import sys

from idrs import *

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='IDR binary file[s] to read')
args = parser.parse_args()
files = args.files

idr = np.dtype(IDRD)

fout = 'idryear.out'

#----------------------------------------------------------------

MJD85 = 46066. # 1985 January 1, 00:00:00 hours

def main():

    f = open(fout, 'w')

    print 'reading files: %d... ' % len(files)

    # iterate over input files
    nfiles = 0
    nptsread = 0
    nptsvalid = 0
    firstfile = True
    for ifname in files:
        # load the whole IDR file in-memory <-- careful with this!
        IDRs = np.fromfile(ifname, dtype=idr, count=-1)

        # iterate over records (IDR structs)
        ybeg = np.inf 
        yend = 0 
        npts = 0
        nrec = IDRs.shape[0]
        for i in xrange(nrec):
            id = IDRs['id'][i]

            # IDR Rev Record
            if id == 'IR':
                #------------------------------------------------
                mjd = IDRs['lat'][i]             # days (integer part)
                secrev = IDRs['lon'][i]          # secs (integer part) 
                fsecrev = IDRs['surf'][i] / 1e6  # secs (fractional part)
                #------------------------------------------------

            # IDR Data Record
            elif id == 'ID': 
                #------------------------------------------------
                secdat = IDRs['time'][i] / 1e6   # secs (since time in Rev)
                surfcheck = IDRs['surf'][i]      # check whether the point is valid
                #------------------------------------------------
                nptsread += 1

		if surfcheck != -9999:
                    nptsvalid += 1

                    # fraction of a day
                    fday = (secrev + fsecrev + secdat) / 86400.
                    MJD = mjd + fday

                    year, month, day, hour, min, sec = mjd2date(MJD)

                    if year < 1970 or year > 2020:  # check possible error
                        continue
                    if ybeg > year:
                        ybeg = year
                    elif yend < year:
                        yend = year

        if ybeg == yend:
            if firstfile is True:
                f.write('%.0f %.0f\n' % (ybeg, yend))
                firstfile = False
            f.write('%s ' % ifname)
            nfiles += 1
        else:
            f.write('\n\n')
            if ybeg != np.inf:
                f.write('%.0f %.0f\n' % (ybeg, yend))
            else:
                f.write('No data points!\n')
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
