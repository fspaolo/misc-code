import os
import sys 
import numpy as np

files = sys.argv[1:]
if len(files) < 1:
    print 'usage: python %s infiles.txt' % sys.argv[0] 
    sys.exit()

print 'processing ...'
for f in files:
    #data = np.loadtxt(f)
    data = np.load(f)
    np.save(os.path.splitext(f)[0]+'.npy', data)
print 'done!'
