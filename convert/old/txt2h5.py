"""
 Convert raw ASCII files to 2D array in HDF5 format. 

 Fernando Paolo <fpaolo@ucsd.edu>
 January 1, 2010
"""

import numpy as np
import tables as tb 
import os
import sys

files = sys.argv[1:]
if len(files) < 1:
    print 'usage: python %s infiles.txt' % sys.argv[0] 
    sys.exit()

print 'converting files: %d... ' % len(files)

for f in files:
    data = np.loadtxt(f)
    h5f = tb.openFile(os.path.splitext(f)[0] + '.h5', 'w')
    h5f.createArray(h5f.root, 'data', data)
    h5f.close()
    
print 'done!'
