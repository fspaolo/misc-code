import os
import sys
import re

# for naming use `keyNumber`: 'geoerm_y1989_m01_r89_t1.h5'

files = sys.argv[1:]
if len(files) < 1 or '-h' in files:
    print 'usage: python %s file1.h5 file2.h5 ...' % sys.argv[0]
    sys.exit()

month = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06',
         'Jul':'07', 'Aug':'08', 'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}

print 'reading files:', len(files)

#for fname in files:
#    for k in month.keys():
#        if k in fname: 
#            newname = fname.replace(k, month[k])
#            os.rename(fname, newname)
#            fname = newname

# sort files by `pattern`
p = lambda s: re.findall('y\d+', s)[0] + re.findall('m\d+', s)[0][-2:]
files.sort(key=p)

# return only the digits
d = lambda s: ''.join([k for k in s if k.isdigit()])

for i, fname in enumerate(files):

    # get and generate new name 
    name = os.path.splitext(os.path.basename(fname))[0]
    sat = name.split('_')[0]
    year, month, reg = re.findall('[ymr]\d+', name)
    newname = '_'.join([sat, 'y'+d(year), 'm'+d(month), 'r'+d(reg), 't%d'%i])

    # rename the file
    newfname = fname.replace(name, newname)
    os.rename(fname, newfname)
    #print 'new fname:', newfname

print 'done!'
