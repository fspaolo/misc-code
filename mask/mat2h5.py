import numpy as np
import tables as tb
import scipy.io as io
import sys
import os

maskfile = sys.argv[1]

print 'loading mask file:', maskfile, '...'
mfile = io.loadmat(maskfile, squeeze_me=True, struct_as_record=True)
MASK = mfile['MASK'].item()

x = MASK[0]
y = MASK[1]
slat = np.array([MASK[2]])
slon = np.array([MASK[3]])
hemi = np.array([str(MASK[4])])
flags = np.array([str(MASK[5])])
date = np.array([str(MASK[6])])
mask = MASK[7]

data = {'x':x, 'y':y, 'mask':mask, 'slat':slat, 'slon':slon,
        'hemi':hemi, 'flags':flags, 'date':date}

f = tb.openFile(os.path.splitext(maskfile)[0]+'.h5', 'w')
filters = tb.Filters(complib='blosc', complevel=5)

for key, value in data.iteritems():
    atom = tb.Atom.from_dtype(value.dtype)
    val = f.createCArray(f.root, key, atom=atom, shape=value.shape, filters=filters)
    val[:] = value

f.close()
