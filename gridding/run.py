# DEPRECATED. x2grid.py now uses 'glob'
"""
Script to run 'x2grid.py' in series on several individual files.

'run.py' accepts two arguments (strings): 'cmd' and 'files'

Example
-------
$ python run.py "python x2grid.py -d .75 .25 -n 10" \
                "/data/alt/ra/envi/hdf/antarctica/xovers/envi_*_tide.h5"
"""

import os
import sys
from glob import glob

cmd = sys.argv[1]
files = glob(sys.argv[2])
for f in files:
    execute = cmd + ' ' + f + ' >> x2grid.log'
    print execute
    os.system(execute)
print 'done.'
