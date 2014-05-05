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
parser.add_argument('-s', dest='suffix', default='', 
        help='suffix for the output filename [default: none]')  

args = parser.parse_args()


class Input(object):
    """
    [TO EDIT] Handles the program input.

    - Get specific variables from specific file type and structure
    - Prepare specific output for the processing
    """
    def __init__(self):
        pass

    def read_config_file(self):
        # search for config file in the same dir as this script
        fname_conf = os.path.realpath(__file__).replace('.py', '.conf')
        if not os.path.exists(fname_conf):
            raise IOError('configuration file not found: %s' % fname_conf)

        # set GMT's x2sys dir to same dir as this script (.conf dir)
        os.environ['X2SYS_HOME'] = os.path.split(fname_conf)[0]

        # parse config file arguments
        self.c = {}
        conf = co.ConfigObj(fname_conf)
        self.c['x2sys_init'] = conf['path']['x2sys_init']
        self.c['x2sys_cross'] = conf['path']['x2sys_cross']
        self.c['tagname'] = conf['init']['tagname']
        self.c['deffile'] = '-D' + conf['init']['deffile']
        self.c['extension'] = '-E' + conf['init']['extension']
        self.c['overwrite'] = '-' + conf['init']['overwrite']
        self.c['distcalc'] = '-C' + conf['init']['distcalc']
        self.c['coordinates'] = '-G' + conf['init']['coordinates']
        self.c['distunits'] = '-Nd' + conf['init']['distunits']
        self.c['maxgap'] = '-Wd' + conf['init']['maxgap']
        self.c['region'] = '-R' + conf['init']['region']
        self.c['coe'] = '-Q' + conf['cross']['coe']
        self.c['interpolation'] = '-I' + conf['cross']['interpolation']
        self.c['values'] = '-' + conf['cross']['values']
        return self.c

    def dbase_init(self):
        """
        Generate data base with given settings (in .conf).
        """
        params = (self.c['x2sys_init'], self.c['tagname'], self.c['deffile'], 
            self.c['extension'], self.c['overwrite'], self.c['distcalc'], 
            self.c['coordinates'], self.c['distunits'], self.c['maxgap'])
        execute = 'sudo %s %s %s %s %s %s %s %s %s -V' % params
        os.system(execute)
        try:
            print 'copy dbase to current dir:'
            os.system('cp -rfv /opt/local/share/gmt/x2sys/%s .' % self.c['tagname'])
        except:
            print 'dbase not copied to current dir!'
            print "couldn't find: /opt/local/share/gmt/x2sys/%s" % self.c['tagname']
            pass
        print 'done.'
        sys.exit()

    def fname_without_ext(self, fname1, fname2, suff=''):
        """
        The name of the output crossover file.
        """
        path2 = os.path.split(fname2)[0]
        fname1, _ = os.path.splitext(os.path.basename(fname1))
        sat = fname1.split('_')[0]
        t1 = re.search('\d\d\d\d\d\d+', fname1).group()
        t2 = re.search('\d\d\d\d\d\d+', fname2).group()
        n1 = re.search('_\d\d_', fname1).group()
        n2 = re.search('_\d\d_', fname2).group()
        if n1 != n2:
            print 'files are from different regions:', fname1, fname2
            sys.exit()
        if suff: suff += '_'
        if '_a' in fname1 and '_d' in fname2:
            suffix = suff + 'ad'
        elif '_d' in fname1 and '_a' in fname2:
            suffix = suff + 'da'
        elif '_d' in fname1 and '_d' in fname2:
            suffix = suff + 'dd'
        elif '_a' in fname1 and '_a' in fname2:
            suffix = suff + 'aa'
        else:
            suffix = ''
        if n1:
            suffix += n1
        if suffix:
            fname_out = '_'.join([sat, t1, t2, suffix])
        else:
            fname_out = '_'.join([sat, t1, t2])
        return os.path.join(path2, fname_out)


def saveh5(fname, data):
    fout = tb.openFile(fname, 'w')
    shape = data.shape
    atom = tb.Atom.from_dtype(data.dtype)
    filters = tb.Filters(complib='blosc', complevel=9)
    dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                             filters=filters)
    dout[:] = data[:] 
    fout.close()

#------------------------------------------------------------------------

def main(args):

    file_ref = args.reffile
    files_in = args.files
    skiprows = args.skiprows
    usecols = args.cols
    suff = args.suffix
    init = args.init

    if not init and not files_in:
        raise IOError("x2sys.py: no input files")

    if not init and file_ref is None and len(files_in) < 2:
        raise IOError("x2sys.py: without `-r` option input files must be > 1")

    if not init and file_ref is None:
        file_ref = files_in[0]    # first file of the list
        files_in.remove(file_ref) 

    # remove reffile from the input file list to avoid crossing 
    # reffile w/itself. For this purpose use `internal crossovers`
    if file_ref in files_in: 
        files_in.remove(file_ref) 
        print 'removing ref file from the input list!'

    In = Input()
    config = In.read_config_file()
    if init: In.dbase_init()

    x2sys_cross = config['x2sys_cross']
    tagname = config['tagname']
    coe = config['coe']
    interpolation = config['interpolation']
    values = config['values']
        
    print 'files to cross:', len(files_in) + 1

    nfiles = 0
    nxovers = 0
    fname_out = None
    for file_i in files_in:

        fname_out = In.fname_without_ext(file_ref, file_i, suff)

        #-------------------------------------------------------------

        params = (x2sys_cross, file_ref, file_i, tagname, 
                  coe, interpolation, values, fname_out) 
        execute = '%s %s %s -T%s %s %s %s -V > %s' % params
        print execute, '...'
        os.system(execute)

        #-------------------------------------------------------------

        print 'file header:'
        os.system('head -n4 %s' % fname_out)
        try:
            data = np.loadtxt(fname_out, skiprows=skiprows, usecols=usecols)
            ##### exclude xovers with NaN values ##### <<<<<<<<<<< later on!
            ##### ind, = np.where(np.isnan(data))
            #np.savetxt(fname_out + '.txt', data, fmt='%.6f')
            saveh5(fname_out + '.h5', data)
            nxovers += data.shape[0]
            nfiles += 1
            print 'number of crossovers:', data.shape[0]
        except:
            fname_out = None
            print 'no crossovers found!'
            print 'files:', file_ref, file_i
            #raise  # shows the error for debuggin

        #os.remove(fname_out)  # test this first. File may be still open!!!

    print 'done.'
    print 'total crossovers:', nxovers
    print 'crossover files created:', nfiles
    if fname_out is not None:
        print 'last output file:', fname_out + '.h5'
    print '\n'

if __name__ == '__main__':
    main(args)
