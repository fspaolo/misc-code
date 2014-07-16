import os
import sys
from glob import glob

arg1 = sys.argv[1]
arg2 = sys.argv[2]
files = glob(arg1)

print 'syncing {0} files ...'.format(len(files))
for f in files:
    os.system('rsync -a {0} {1}'.format(f, arg2))
print 'done.'

