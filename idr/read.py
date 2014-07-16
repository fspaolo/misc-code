from glob import glob
import os

h5f = glob('/Users/fpaolo/data/altim/ERS2/*.h5')
txtf = glob('/Users/fpaolo/data/altim/ERS2/*.txt')

for f in txtf:
    path, file = os.path.split(f)
    fname, ext = os.path.splitext(file)
    if (f[:-3]+'h5') not in h5f:
        os.system('./readidr_ra1 /Volumes/LaCie1TB/ANTARCTICA/ERS2/%s' % fname)
        os.system('cp /Volumes/LaCie1TB/ANTARCTICA/ERS2/%s %s' % (file, path))
        print file, '->', path
