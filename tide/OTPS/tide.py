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
parser.add_argument('-k', dest='skiprows', type=int, default=4,
    help='skip the first SKIPROWS lines from output file [default: 4]')
parser.add_argument('-o', dest='cols', nargs='*', default= \
    [0,1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    help='which columns to output (-o 0 1 6 7 ..)')
parser.add_argument('-s', dest='suffix', default='', 
        help='suffix for the output filename [default: none]')  

args = parser.parse_args()


class Input(object):
    """
    [TO EDIT] Handles the program input.

    - Get specific variables from specific file type and structure
    - Prepare specific output for the processing
    """
    def __init__(self, fname_in):
        self.fname_in = fname_in

    def read_config_file(self):
        # search for config file in the same dir as this script
        fname_conf = os.path.realpath(__file__).replace('.py', '.conf')
        if not os.path.exists(fname_conf):
            raise IOError('configuration file not found: %s' % fname_conf)

        # parse config file arguments
        self.c = {}
        conf = co.ConfigObj(fname_conf)
        self.c['elev_model'] = conf['model']['elev_model']
        self.c['transp_model'] = conf['model']['transp_model']
        self.c['bathy_grid'] = conf['model']['bathy_grid']
        self.c['convert_func'] = conf['model']['convert_func']
        # ...

    def gen_control_file(self):
        path, name = os.path.split(self.c['elev_model'])
        self.fname = os.path.join(path, 'Model_' + os.path.splitext(name)[0])
        fid = open(self.fname, 'w')
        fid.write(self.c['elev_model'] + '\n')
        fid.write(self.c['transp_model'] + '\n')
        fid.write(self.c['bathy_grid'] + '\n')
        fid.write(self.c['convert_func'] + '\n')
        fid.close()

    def gen_setup_file(self):

        fname_setup = os.path.splitext(self.fname_in)[0] + '.inp'
        fid = open(fname_setup, 'w')
        fid.write('TEST')
        fid.close()

    def gen_llt_file(self):
        pass


    def fname_without_ext(self, fname1, fname2, suff=''):
        """The name of the output crossover file."""
        path2 = os.path.split(fname2)[0]
        fname1, _ = os.path.splitext(os.path.basename(fname1))
        sat, t1 = fname1.split('_')[:2]
        t2 = re.search('\d\d\d\d\d\d+', fname2).group()
        if suff: suff += '_'
        if '_asc' in fname1 and '_des' in fname2:
            suffix = suff + 'ad'
        elif '_des' in fname1 and '_asc' in fname2:
            suffix = suff + 'da'
        elif '_des' in fname1 and '_des' in fname2:
            suffix = suff + 'dd'
        elif '_asc' in fname1 and '_asc' in fname2:
            suffix = suff + 'aa'
        else:
            suffix = ''
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

    files_in = args.files
    skiprows = args.skiprows
    usecols = args.cols
    suff = args.suffix

    print 'files to cross:', len(files_in) + 1

    nfiles = 0
    nxovers = 0
    fname_out = None
    for ifile in files_in:

        In = Input(ifile)
        In.read_config_file()
        In.gen_control_file()
        In.gen_setup_file()
        sys.exit()

        # ...

        #-------------------------------------------------------------

        print 'predict_tide: processing', ifile, '...' 
        os.system('%s < %s' % (predict_tide, setup_file))

        #-------------------------------------------------------------

        #print 'file header:'
        #os.system('head -n4 %s' % fname_out)
        try:
            data = np.loadtxt(fname_out, skiprows=skiprows, usecols=usecols)
            #np.savetxt(fname_out + '.txt', data, fmt='%.6f')
            saveh5(fname_out + '.h5', data)
            nxovers += data.shape[0]
            nfiles += 1
            print 'number of crossovers:', data.shape[0]
        except:
            fname_out = None
            print 'no crossovers found!'
            #raise  # show the error

        #os.remove(fname_out)  # test this first. File may be still open!!!

    print 'done.'
    print 'total crossovers:', nxovers
    print 'crossover files created:', nfiles
    if fname_out is not None:
        print 'last output file:', fname_out + '.h5'
    print '\n'

if __name__ == '__main__':
    main(args)
