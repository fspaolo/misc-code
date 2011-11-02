"""
Script to automatize the input/output of GMT's x2sys program.

Read several HDF5/ASCII files and cross them with a given *reference* 
file. For example, for a `reffile_0.ext` and `infile_1.ext, infile_2.ext, 
infile_3.ext, ...`, we would have:

    reffile0.ext - infile1.ext, 
    reffile0.ext - infile2.ext, 
    reffile0.ext - infile3.ext, 
    ... 

Note: command line options for the `x2sys` program are automatically read 
from a configuration file `x2sys.ini` that can be edited.

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# November 8, 2010

import os
import sys
import argparse as ap
import configobj as co
import numpy as np
import tables as tb
import tempfile as tf

# parse configuration file 
config = co.ConfigObj('/Users/fpaolo/code/x2sys/x2sys.ini')
tagname = config['init']['tagname']
deffile = '-D' + config['init']['deffile']
extension = '-E' + config['init']['extension']
overwrite = '-' + config['init']['overwrite']
distcalc = '-C' + config['init']['distcalc']
coordinates = '-G' + config['init']['coordinates']
distunits = '-Nd' + config['init']['distunits']
maxgap = '-Wd' + config['init']['maxgap']
region = '-R' + config['init']['region']
coe = '-Q' + config['cross']['coe']
interpolation = '-I' + config['cross']['interpolation']
values = '-' + config['cross']['values']

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='*', help='HDF5/ASCII file[s] to read')
parser.add_argument('-r', dest='reffile', default=None, 
    help='reference file to cross w/others: reffile - infiles')  
parser.add_argument('-k', dest='skiprows', type=int, default=4,
    help='skip the first SKIPROWS lines from output file [default: 4]')
parser.add_argument('-o', dest='cols', nargs='*', default= \
    [0,1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    help='which columns to output (-o 0 1 6 7 ..)')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='ASCII input/output files [default: HDF5]')
parser.add_argument('-i', dest='init', default=False, action='store_const',
    const=True, help='initialize x2sys data base for track data files '
    '[default: no]')

args = parser.parse_args()

if not args.init and args.reffile is None:
    print "x2sys.py: error: argument -r is required"
    sys.exit()
if not args.init and not args.file:
    print "x2sys.py: error: no input files"
    sys.exit()

files = args.file
file_ref = args.reffile
skiprows = args.skiprows
usecols = args.cols
ascii = args.ascii


# remove reffile from the input file list
# to avoid crossing reffile w/itself
if file_ref in files: 
    #files.remove(file_ref) 
    print 'crossing reference file with it self!!!'
    if not files:
        print 'ref file:', file_ref
        print 'not crossing ref file w/itself!'
        sys.exit()


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
    filters = tb.Filters(complib='blosc', complevel=5)
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


def ofname_without_ext(fname1, fname2):
    """The name of output crossover file.
    """
    pathfname1, ext1 = os.path.splitext(fname1)
    pathfname2, ext2 = os.path.splitext(fname2)
    path1, fname1 = os.path.split(pathfname1)
    path2, fname2 = os.path.split(pathfname2)
    return os.path.join(path2, fname1 + '-' + fname2)

#------------------------------------------------------------------------

# generate data base with given settings (x2sys.ini) 
if args.init:
    #os.system('sudo rm -rfv /opt/local/share/gmt/x2sys/ALTIM')
    os.system('sudo x2sys_init %s %s %s %s %s %s %s %s -V' % \
              (tagname, deffile, extension, overwrite, distcalc, \
               coordinates, distunits, maxgap))
    sys.exit()

    
print 'files to cross:', len(files) + 1

nfiles = 0
nxovers = 0
for file_i in files:

    file_out = ofname_without_ext(file_ref, file_i)

    print 'x2sys_cross: processing', file_ref, '-', file_i, '...'
    os.system('/opt/local/bin/x2sys_cross %s %s -T%s %s %s %s > %s' % \
              (file_ref, file_i, tagname, coe, interpolation, values, file_out))

    #print 'file header:'
    #os.system('head -n4 %s' % file_out)
    try:
        data = np.loadtxt(file_out, skiprows=skiprows, usecols=usecols)
        np.savetxt(file_out + '.txt', data, fmt='%.6f')
        saveh5(file_out + '.h5', data)
        nxovers += data.shape[0]
        nfiles += 1
        print 'number of crossovers:', data.shape[0]
    except:
        print 'no crossovers found!'
        #raise  # shows the error
    

print 'done!'
print 'total crossovers:', nxovers
print 'crossover files created:', nfiles
print 'last output file:', file_out + '.h5'
