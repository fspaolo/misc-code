"""
 Convert ASCII data files to HDF5 format (2D array) and vice-versa.

 The program recognizes the type of inputed files (ASCII or HDF5) 
 and perform the conversion accordingly.

 Fernando Paolo <fpaolo@ucsd.edu>
 January 1, 2010
"""

import numpy as np
import tables as tb 
import os
import sys

files = sys.argv[1:]
if len(files) < 1:
    print 'usage: (TXT to HDF5) python %s file1.txt file2.txt ...' % sys.argv[0] 
    print '       (HDF5 to TXT) python %s file1.h5 file2.h5 ...' % sys.argv[0] 
    sys.exit()

print 'converting files: %d... ' % len(files)

try:
    for f in files:
        data = np.loadtxt(f)
        h5f = tb.openFile(os.path.splitext(f)[0] + '.h5', 'w')
        h5f.createArray(h5f.root, 'data', data)
        h5f.close()
except:
    print 'PASS1'
    pass

try:
    for f in files:
        h5f = tb.openFile(f, 'r')
        data = h5f.root.data.read()
        h5f.close()
        np.savetxt(os.path.splitext(f)[0] + '.txt', data, fmt='%f')
except:
    print 'PASS2'
    pass

else:
    print 'Error!'

finally:
   print 'End!'

    
print 'done!'
