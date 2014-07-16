import os
import sys 
import numpy as np

files = sys.argv[1:]
print 'processing ...'
for f in files:
    data = np.load(f)
    np.savetxt(os.path.splitext(f)[0]+'.txt', data)
print 'done!'
