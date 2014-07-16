import numpy as np
import pylab as pl

pl.figure()

#d1 = np.loadtxt('ers1_199407.txt')
#d2 = np.loadtxt('seasat_197810.txt')
d3 = np.loadtxt('ers1_199407-seasat_197810.txt')

#i, = np.where(d3[:,0] < 0)
#d3[i,0] += 360

#pl.plot(d1[:,3], d1[:,2], '.')
#pl.plot(d2[:,3], d2[:,2], '.')
pl.plot(d3[:,0], d3[:,1], '.')

'''
pl.figure()

d1 = np.loadtxt('seasat_197809.txt')
d2 = np.loadtxt('seasat_197810.txt')
d3 = np.loadtxt('seasat_197809-seasat_197810.txt2')

i, = np.where(d3[:,0] < 0)
d3[i,0] += 360

pl.plot(d1[:,3], d1[:,2], '.')
pl.plot(d2[:,3], d2[:,2], '.')
pl.plot(d3[:,0], d3[:,1], '.')
'''
pl.show()
