import numpy as np
import tables as tb
import pylab as pl
import sys

'''
f = tb.openFile('all_200101_des.h5')
a = f.root.data.read()
f.close()

f = tb.openFile('all_200101_des.h5_orig')
b = f.root.data.read()
f.close()

print a[0,:]

print 'A :', a.shape, ', B :', b.shape

print 'A == B :', np.alltrue(a == b)

'''

f = tb.openFile(sys.argv[1])
d = f.root.data
i1, = np.where(d[:,-1] == 0)
i2, = np.where(d[:,-1] == 1)
pl.plot(d[i1,3], d[i1,2], '.')
pl.plot(d[i2,3], d[i2,2], '.')
pl.show()
