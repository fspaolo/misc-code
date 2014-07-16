import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
from masklib import *

f = tb.openFile(sys.argv[1])

lon = f.root.idr.cols.lon[:]
lat = f.root.idr.cols.lat[:]
fmask = f.root.mask1.cols.fmask[:]
fborder = f.root.mask1.cols.fborder[:]
fbuff = f.root.mask1.cols.fbuff[:]

x, y = ll2xy(lon, lat)

'''
lon += 70
lon[lon>=360] -= 360.
lon[lon<0] += 360.
'''
ii, = np.where(((fmask == 0) | (fmask == 1)) & (fbuff == 1))
#plt.plot(x[fmask==4], y[fmask==4], '.')
plt.plot(x[ii], y[ii], '.')
#plt.plot(lon, lat, '.')
plt.show()

f.close()
