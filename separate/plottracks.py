import numpy as np
import pylab as pl
import tables as tb
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file to read')
parser.add_argument('-x', dest='loncol', default=3, type=int,
                    help='column of longitude in the file (0,1,..) [3]')  
parser.add_argument('-y', dest='latcol', default=2, type=int,
                    help='column of latitude in the file (0,1,..) [2]')  
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
                    const=True, help='reads and writes ASCII file [default HDF5]')

args = parser.parse_args()
files = args.file
loncol = args.loncol
latcol = args.latcol
ascii = args.ascii

for f in files:
    print 'loading file ...'
    if ascii:
        data = np.loadtxt(f)
    else:
        h5f = tb.openFile(f)
        data = h5f.root.data.read()
        h5f.close()
     
    print 'plotting file ...'
    #pl.plot(data[:,loncol], data[:,latcol], '.')
    ia, = np.where(data[:,-1] == 0)
    id, = np.where(data[:,-1] == 1)
    ic, = np.where((data[:,-1] != 0) & (data[:,-1] != 1))
    pl.plot(data[ia,loncol], data[ia,latcol], 'b.')
    pl.plot(data[id,loncol], data[id,latcol], 'g.')
    pl.plot(data[ic,loncol], data[ic,latcol], 'ro')

pl.show()
