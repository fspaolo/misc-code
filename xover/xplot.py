#!/usr/bin/python

import pylab as pl

f1 = pl.loadtxt('3850409AC.ID04.txt')
f2 = pl.loadtxt('3860809AC.ID04.txt')
x = pl.loadtxt('xover.out')
for i in xrange(len(x)):
    if x[i,1] > 180:
        x[i,1] -= 360.

#c1 = pl.loadtxt('/Users/fpaolo/helen/moa_coastfile_ll.txt')
#c2 = pl.loadtxt('/Users/fpaolo/helen/moa_islands_ll.txt')
#c3 = pl.loadtxt('/Users/fpaolo/helen/antarctica_gl_ll.txt')

#pl.plot(c1[:,1], c1[:,0])
#pl.plot(c2[:,1], c2[:,0])
#pl.plot(c3[:,1], c3[:,0])

pl.plot(f1[:,3], f1[:,2], '.')
pl.plot(f2[:,3], f2[:,2], '.')
pl.plot(x[:,1], x[:,0], 'ro')

pl.legend(('track set 1', 'track set 2', 'crossovers'), loc=2)
pl.title('program: xover')
pl.savefig('xover.png')
pl.show()
