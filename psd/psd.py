import numpy as np
import matplotlib.pyplot as pl

def ticks(ax):
    for loc, spine in ax.spines.iteritems():
        if loc in ['left', 'bottom']:
            spine.set_position(('outward',10)) # outward by 10 points
        elif loc in ['right','top']:
            spine.set_color('none')            # don't draw spine
        else:
            raise ValueError('unknown spine location: %s'%loc)
    # turn off ticks where there is no spine
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

S = lambda A, B, k: A * np.exp(-B * k)

#-----------------------------------------------------------

'''
mag = np.loadtxt('mag.dat')
fig = pl.figure()
ax = fig.add_subplot(1,1,1)
ticks(ax)

pl.plot(mag)
pl.xlim(0, 2000)
pl.ylim(-300, 300)
pl.ylabel('Mag, nT')
pl.xlabel('Distance, km')
pl.legend(('Magnetic field',)).draw_frame(False)
'''

###

fig = pl.figure()
ax = fig.add_subplot(1,1,1)
ticks(ax)

freq, psd, err, res, ntapers = np.loadtxt('psd1.dat', unpack=True)
pl.plot(freq, psd, linewidth=1.5)
freq, psd, err, res, ntapers = np.loadtxt('psd2.dat', unpack=True)
pl.plot(freq, psd, linewidth=1.5)

k = np.linspace(0, 0.5, 1000)
pl.plot(k, S(1500, 10, k), linewidth=1)

pl.title('with prewhitening')
pl.ylabel('PSD, nT$^2$km')
pl.xlabel('Frequency, km$^{-1}$')
pl.legend(('sine multitaper', 'prolate spheroidal', \
           'theoretical value')).draw_frame(False)

###

'''
fig = pl.figure()
ax = fig.add_subplot(3,1,1)
ticks(ax)
pl.plot(freq, err)
pl.ylabel('Error, $\pm \sigma$')

ax = fig.add_subplot(3,1,2)
ticks(ax)
pl.plot(freq, res)
pl.ylabel('Resolution')

ax = fig.add_subplot(3,1,3)
ticks(ax)
pl.plot(freq, ntapers)
pl.ylabel('N Tapers')
pl.xlabel('Frequency, km$^{-1}$')
'''

#pl.legend(('Magnetic field',)).draw_frame(False)
pl.show()
