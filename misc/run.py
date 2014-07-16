import os
from glob import glob

files = glob('/data/alt/tmp/*.HDF')
files = ' '.join(files)
os.system('/Library/Frameworks/EPD64.framework/Versions/Current/bin/python gla2idr.py {0}'.format(f))
