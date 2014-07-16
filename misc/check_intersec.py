"""
Usage: python check_inter.py '/path/to/files1.ext' '/path/to/files2.ext'
"""

import sys
import numpy as np
from glob import glob

f1 = np.array(glob('%s' % sys.argv[1]))
f2 = np.array(glob('%s' % sys.argv[2]))

f11 = np.array([f.split('.')[0] for f in f1])
f22 = np.array([f.split('.')[0] for f in f2])

a = np.in1d(f11, f22)
i, = np.where(a==False)

print 'files_1 not in files_2:'
print ' '.join(f1[i])
