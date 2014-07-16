#!/usr/bin/env python

import pylab as P

file = 'trackfound.out'
lon, lat = P.load(file, usecols=(0,1), unpack=True)
P.plot(lon, lat, '.')
#P.plot(time, h)
#P.xlim(1.0e8, 1.1e8)
P.axis([318, 326, -6, 8])
P.show()

