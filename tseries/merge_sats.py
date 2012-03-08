#!/usr/bin/env python
"""

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# March 7, 2012

import sys
import argparse as ap

sys.path.append('/Users/fpaolo/code/misc')
from util import *

# global variables
#-------------------------------------------------------------------------

SAVE_TO_FILE = True
NODE_NAME = 'fris'

#-------------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file_in', nargs='+', 
    help='HDF5 files with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: /input/path/all_t1_t2_grids_mean.h5]')

args = parser.parse_args()


def get_output_dimensions(fnames, node_name):
    N= 0
    for fname in fnames:
        f = tb.openFile(fname, 'r')
        data = f.getNode('/', node_name)
        N += data.table.nrows
        ny, = data.lat.shape
        nx, = data.lon.shape
        f.close()
    return (N, ny, nx)


def create_output_containers(fname_out, any_data, shape, node_name, chunkshape):
    # open or create output file
    file_out = tb.openFile(fname_out, 'w')
    filters = tb.Filters(complib='blosc', complevel=9)
    atom = tb.Atom.from_dtype(any_data.dtype)
    N, ny, nx = shape
    cs = chunkshape 
    title = ''
    dout = {}
    g = file_out.createGroup('/', node_name)
    dout['table'] = file_out.createTable(g, 'table', 
                                 TimeSeriesGrid, title, filters)
    dout['lon'] = file_out.createCArray(g, 'lon', 
                                  atom, (nx,), '', filters)
    dout['lat'] = file_out.createCArray(g, 'lat', 
                                  atom, (ny,), '', filters)
    dout['x_edges'] = file_out.createCArray(g, 'x_edges', 
                                  atom, (nx+1,), '', filters)
    dout['y_edges'] = file_out.createCArray(g, 'y_edges', 
                                  atom, (ny+1,), '', filters)
    dout['dh_mean'] = file_out.createCArray(g, 'dh_mean', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dh_mean_corr'] = file_out.createCArray(g, 'dh_mean_corr', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dh_error'] = file_out.createCArray(g, 'dh_error', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dg_mean'] = file_out.createCArray(g, 'dg_mean', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dg_error'] = file_out.createCArray(g, 'dg_error', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['n_ad'] = file_out.createCArray(g, 'n_ad', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['n_da'] = file_out.createCArray(g, 'n_da', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    return dout, file_out


def get_fname_out(files, fname_out=None, prefix=None, sufix=None):
    """
    Construct the output file name with the min and max times 
    from the input files.
    """
    path, name = os.path.split(files[0])  # path from any file
    if fname_out is None:
        times = [re.findall('\d\d\d\d\d\d+', fname) for fname in files]
        t_1 = [t1 for t1, t2 in times]
        t_2 = [t2 for t1, t2 in times]
        t_min, t_max =  min(t_1), max(t_2)
        if prefix is None:
            prefix = name.split('_')[0]  # sat name
        name = '_'.join([prefix, t_min, t_max, sufix])
    else:
        name = fname_out
    return os.path.join(path, name)


def main(args):

    fnames_in = args.file_in
    fname_out = args.fname_out

    n1, n2 = 0, 0

    isfirst = True
    for fname in fnames_in:

        # input data
        #-------------------------------------------------------------

        d = {}
        file_in = tb.openFile(fname)
        data = file_in.getNode('/', NODE_NAME)

        d['table'] = data.table
        d['dh_mean'] = data.dh_mean
        d['dh_mean_corr'] = data.dh_mean_corr
        d['dh_error'] = data.dh_error
        d['dg_mean'] = data.dg_mean
        d['dg_error'] = data.dg_error
        d['n_ad'] = data.n_ad
        d['n_da'] = data.n_da
        d['x_edges'] = data.x_edges
        d['y_edges'] = data.y_edges
        d['lon'] = data.lon
        d['lat'] = data.lat

        # output data
        #-------------------------------------------------------------

        if not SAVE_TO_FILE: continue

        if isfirst:
            # open or create output file
            isfirst = False
            fname_out = get_fname_out(fnames_in, fname_out=fname_out, 
                                      prefix='all', sufix='grids_mean.h5')
            N, ny, nx = get_output_dimensions(fnames_in, NODE_NAME)
            arr = np.empty(1, 'f8')
            dout, file_out = create_output_containers(
                             fname_out, arr, (N,ny,nx), NODE_NAME, None)

            # save info
            dout['lon'][:] = d['lon'][:]
            dout['lat'][:] = d['lat'][:]
            dout['x_edges'][:] = d['x_edges'][:]
            dout['y_edges'][:] = d['y_edges'][:]

        n2 += data.table.nrows
        print N, n1, n2

        dout['table'].append(d['table'][:])
        dout['table'].flush()
        dout['dh_mean'][n1:n2,...] = d['dh_mean'][:]
        dout['dh_mean_corr'][n1:n2,...] = d['dh_mean_corr'][:]
        dout['dh_error'][n1:n2,...] = d['dh_error'][:]
        dout['dg_mean'][n1:n2,...] = d['dg_mean'][:]
        dout['dg_error'][n1:n2,...] = d['dh_error'][:]
        dout['n_ad'][n1:n2,...] = d['n_ad'][:]
        dout['n_da'][n1:n2,...] = d['n_da'][:] 

        n1 = n2
        file_in.close()

    file_out.flush()
    file_out.close()

    print 'out file -->', fname_out


if __name__ == '__main__':
    main(args)
