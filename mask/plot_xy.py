import numpy as np
import tables as tb
import pylab as pl
import sys

f = tb.openFile(sys.argv[1])
d = f.root.data

i, = np.where(d[:,-2] == int(sys.argv[2]))

if i.shape[0] > 0:
    pl.plot(d[i,3], d[i,2], '.')
    pl.show()
else:
    print 'no data points!'
