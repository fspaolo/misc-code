"""
Script to automatize the input/output of GMT's x2sys program.

Read several HDF5/ASCII files and cross them with a given *reference* 
file. Example, for a reference `file0.ext` and an input list `file1.ext, 
file2.ext, file3.ext, ...`, we have:

    $ x2sys.py -r file0.ext file1.ext file2.ext file3.ext ...

    with crossover outputs:

    file0.ext - file1.ext, 
    file0.ext - file2.ext, 
    file0.ext - file3.ext, 
    ... 

Note 1
------
If `-r` option is not specified, then the reference file is the first
of the input file list.

Note 2
------
command line options for the `x2sys` program are automatically read 
from a configuration file `x2sys.conf` that can be edited.

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# November 8, 2010

import os
import sys
import re
import argparse as ap
import configobj as co
import numpy as np
import tables as tb
import tempfile as tf


# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='*', help='HDF5/ASCII file[s] to read')
parser.add_argument('-r', dest='reffile', default=None, 
    help='reference file to cross w/others: file_ref - files_in')  
parser.add_argument('-k', dest='skiprows', type=int, default=4,
    help='skip the first SKIPROWS lines from output file [default: 4]')
parser.add_argument('-o', dest='cols', nargs='*', default= \
    [0,1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    help='which columns to output (-o 0 1 6 7 ..)')
parser.add_argument('-i', dest='init', default=False, action='store_const',
    const=True, help='initialize x2sys data base for track data files '
    '[default: no]')
args = parser.parse_args()

file_ref = args.reffile
files_in = args.files
skiprows = args.skiprows
usecols = args.cols
init = args.init

# search for config file in the same dir as this script
fname_conf = os.path.realpath(__file__).replace('.py', '.conf')
if not os.path.exists(fname_conf):
    raise IOError('configuration file not found: %s' % fname_conf)

# set GMT's x2sys dir to same dir as this script (.conf dir)
os.environ['X2SYS_HOME'] = os.path.split(fname_conf)[0]  # not working !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# parse config file arguments
conf = co.ConfigObj(fname_conf)
x2sys_init = conf['path']['x2sys_init']
x2sys_cross = conf['path']['x2sys_cross']
tagname = conf['init']['tagname']
deffile = '-D' + conf['init']['deffile']
extension = '-E' + conf['init']['extension']
overwrite = '-' + conf['init']['overwrite']
distcalc = '-C' + conf['init']['distcalc']
coordinates = '-G' + conf['init']['coordinates']
distunits = '-Nd' + conf['init']['distunits']
maxgap = '-Wd' + conf['init']['maxgap']
region = '-R' + conf['init']['region']
coe = '-Q' + conf['cross']['coe']
interpolation = '-I' + conf['cross']['interpolation']
values = '-' + conf['cross']['values']

if not init and not files_in:
    raise IOError("x2sys.py: no input files")

if not init and file_ref is None and len(files_in) < 2:
    raise IOError("x2sys.py: without `-r` option input files must be > 1")

if not init and file_ref is None:
    file_ref = files_in[0]
    files_in.remove(file_ref) 

# remove reffile from the input file list to avoid crossing 
# reffile w/itself. Use `internal crossovers` for this purpose
if file_ref in files_in: 
    files_in.remove(file_ref) 
    print 'removing ref file from the input list!'


def readh5(fname, inmemory=False):
    fin = tb.openFile(fname, 'r')
    if inmemory:
        data = fin.root.data.read()
    else:
        data = fin.root.data
    return data, fin


def closeh5(fid):
    try:
        if fid.isopen: 
            fid.close()
    except:
        pass


def saveh5(fname, data):
    fout = tb.openFile(fname, 'w')
    shape = data.shape
    atom = tb.Atom.from_dtype(data.dtype)
    filters = tb.Filters(complib='blosc', complevel=9)
    dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                             filters=filters)
    dout[:] = data[:] 
    fout.close()


def hdf2txt_tmp(fname):
    """Create a temporary txt file from HDF5.
    """
    print 'converting HDF5 -> temporary ASCII ...',
    data, fin = readh5(fname)
    f = tf.NamedTemporaryFile(suffix='')  # create temp file
    np.savetxt(f, data, fmt='%f')
    f.seek(0)
    closeh5(fin)
    print 'done'
    return f


def fname_out_no_ext(fname1, fname2):
    """The name of output crossover file: fname1_fname2.
    """
    fname1, _ = os.path.splitext(os.path.basename(fname1))
    path2, fname2 = os.path.split(os.path.splitext(fname2)[0])
    return os.path.join(path2, fname1 + '_' + fname2)


def fname_out_no_ext2(fname1, fname2):
    """The name of output crossover file: formatted.
    """
    path2 = os.path.split(fname2)[0]
    fname1, _ = os.path.splitext(os.path.basename(fname1))
    sat, t1, r1 = fname1.split('_')[:3]
    t2 = re.search('\d\d\d\d\d\d+', fname2).group()
    return os.path.join(path2, '_'.join([sat, t1, t2, r1]))


def fname_out_no_ext3(fname1, fname2):
    """The name of output crossover file: formatted.
    """
    path2 = os.path.split(fname2)[0]
    fname1, _ = os.path.splitext(os.path.basename(fname1))
    sat, t1 = fname1.split('_')[:2]
    t2 = re.search('\d\d\d\d\d\d+', fname2).group()
    if '_asc' in fname1 and '_des' in fname2:
        suffix = 'ad'
    elif '_des' in fname1 and '_asc' in fname2:
        suffix = 'da'
    elif '_des' in fname1 and '_des' in fname2:
        suffix = 'dd'
    elif '_asc' in fname1 and '_asc' in fname2:
        suffix = 'aa'
    else:
        suffix = ''
    if suffix:
        fname_out = '_'.join([sat, t1, t2, suffix])
    else:
        fname_out = '_'.join([sat, t1, t2])
    return os.path.join(path2, fname_out)

#------------------------------------------------------------------------

def main():
    # generate data base with given settings (x2sys.ini) 
    if init:
        #os.system('sudo rm -rfv /opt/local/share/gmt/x2sys/ALTIM')
        params = (x2sys_init, tagname, deffile, extension, overwrite, 
                  distcalc, coordinates, distunits, maxgap)
        execute = 'sudo %s %s %s %s %s %s %s %s %s -V' % params
        os.system(execute)
        sys.exit()
        
    print 'files to cross:', len(files_in) + 1

    nfiles = 0
    nxovers = 0
    file_out = None
    for file_i in files_in:

        #file_out = fname_out_no_ext(file_ref, file_i)
        #file_out = fname_out_no_ext2(file_ref, file_i)
        file_out = fname_out_no_ext3(file_ref, file_i)

        print 'x2sys_cross: processing', file_ref, '-', file_i, '...'
        params = (x2sys_cross, file_ref, file_i, tagname, coe, interpolation, values, file_out) 
        execute = '%s %s %s -T%s %s %s %s > %s' % params
        os.system(execute)

        #print 'file header:'
        #os.system('head -n4 %s' % file_out)
        try:
            data = np.loadtxt(file_out, skiprows=skiprows, usecols=usecols)
            #np.savetxt(file_out + '.txt', data, fmt='%.6f')
            saveh5(file_out + '.h5', data)
            nxovers += data.shape[0]
            nfiles += 1
            print 'number of crossovers:', data.shape[0]
        except:
            file_out = None
            print 'no crossovers found!'
            #raise  # shows the error

        #os.remove(file_out)  # test this first. File may be still open!!!

    print 'done.'
    print 'total crossovers:', nxovers
    print 'crossover files created:', nfiles
    if file_out is not None:
        print 'last output file:', file_out + '.h5'
    print '\n'

if __name__ == '__main__':
    main()
