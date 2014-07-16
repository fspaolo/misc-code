"""
 Convert 2D array HDF5 files to raw ASCII format. 

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
    h5f = tb.openFile(f, 'r')
    data = h5f.root.data.read()
    h5f.close()
    np.savetxt(os.path.splitext(f)[0] + '.txt', data, fmt='%f')
    
print 'done!'
