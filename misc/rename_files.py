import os
import sys
import re
from glob import glob

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
#p = lambda s: re.findall('y\d+', s)[0] + re.findall('m\d+', s)[0][-2:]
#p = lambda s: re.findall('\d+', s)[0] + re.findall('\d+', s)[0][-2:]
#files.sort(key=p)

# return only the digits
d = lambda s: ''.join([k for k in s if k.isdigit()])

def rename1(files):
    for fname in files:

        ### get and generate new name 
        name = os.path.splitext(os.path.basename(fname))[0]
        sat, year, month, reg = name.split('_')[:4]
        #sat, time, reg = name.split('_')[:4]
        time = d(year) + '%02d' % (int(d(month)[-2:]) - 1)
        reg = '%02d' % (int(d(reg)) + 1)
        #reg = '%02d' % (int(d(reg)) - 1)
        newname = '_'.join([sat, time, reg])
        #year, month, reg = re.findall('[ymr]\d+', name)
        #newname = '_'.join([sat, 'y'+d(year), 'm'+d(month), 'r'+d(reg)])

        ### rename the file
        fname_new = fname.replace(name, newname)
        os.rename(fname, fname_new)
        #print 'new fname:', fname_new

def rename2(files):

    onefile = files[0]
    path, _ = os.path.split(onefile)
    fs = os.path.join(path, '*.old')
    files = glob(fs)

    for fname in files:

        ### get pattern to be changed 
        yyyymm = re.findall('_\d\d\d\d01', fname)
        if yyyymm:
            yyyymm = d(yyyymm[0])
        else:
            continue
        yyyy, mm = yyyymm[:4], yyyymm[4:]
        yyyymm_new = str(int(yyyy) + 1) + mm
        yyyymm = '_' + yyyymm
        yyyymm_new = '_' + yyyymm_new

        ### rename the file
        fname_new = fname.replace(yyyymm, yyyymm_new)  # replace *all* occurencies
        fname_new = fname_new.replace('.old', '')
        os.rename(fname, fname_new)
        print 'change:', yyyymm, '-->', yyyymm_new
        print 'old:', fname
        print 'new:', fname_new
        print ''


def rename3(files):
    for fname in files:
        for i in range(100000):
            pass
        fname_new = fname.replace('_fris', '')
        os.rename(fname, fname_new)
        #print 'old:', fname
        #print 'new:', fname_new


#rename1(files)
#rename2(files)
rename3(files)

print 'done.'
