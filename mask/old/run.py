import numpy as np
from glob import glob
import sys
import os

'''
### read IDR -> TXT
idrf = glob('*.ID05')
txtf = glob('*.txt')

txt = [os.path.splitext(f)[0] for f in txtf]

print 'total IDR:', len(idrf)

for t in txt:
    if t in idrf:
        idrf.remove(t)

print 'reading IDR:', len(idrf)

for f in idrf:
    os.system('./readidr_ra1 %s' % f)
'''
'''
### extract region.py 
h5 = glob('*.ID05.h5')
maskf = glob('*_mask.h5')

hh5 = [os.path.splitext(f)[0] for f in h5]
mask = [os.path.splitext(f)[0] for f in maskf]

print 'total h5:', len(h5)

for f in mask:
    f = f[:-5]
    if f in hh5:
        hh5.remove(f)

print 'reading h5:', len(hh5)

for f in hh5:
    os.system('region.py -r 0 360 -73 -62 %s' % (f+'.h5'))
'''
### apply mask
h5 = glob('/Users/fpaolo/data/altim/ENVISAT/*_reg.h5')
maskf = glob('/Users/fpaolo/data/altim/ENVISAT/*_reg_mask.h5')

hh5 = [os.path.splitext(f)[0] for f in h5]
mask = [os.path.splitext(f)[0] for f in maskf]

print 'total h5:', len(h5)

for f in mask:
    f = f[:-5]
    if f in hh5:
        hh5.remove(f)

print 'reading h5:', len(hh5)

for f in hh5:
    os.system('python runmask.py %s' % (f+'.h5'))
