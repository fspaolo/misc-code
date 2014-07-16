import numpy as N
import spacing    # Fortran 90 module

file = '/home/fspaolo/DATA/work/norte/bat/tracks/equant.txt'

#XY = N.loadtxt(file)
x,y = N.loadtxt(file, unpack=True, usecols=(0,1))
XY = N.column_stack((x, y))
n = len(XY)

dist = spacing.mean_dist(XY, n)  # Fortran subroutine
dist_km = dist*111.0

print 'mean dist:\t%g km' % dist_km
print 'resolution:\t%g km' % (2*dist_km)
