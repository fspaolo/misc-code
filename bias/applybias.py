#!/usr/bin/env python
"""
Applies ICESat pampaign biases to crossover grids. 

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# November 8, 2012

import sys
import numpy as np
import tables as tb
import argparse as ap

# parameters 

H_NAME = 'dh_mean'
SAVE_AS_NAME = 'dh_mean_u'

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs=1, 
    help='HDF5 file with grids to read (3D arrays)')
parser.add_argument('-b', dest='source', default=None,
    help='apply intercampaign biases (urban/zwally/borsa), default none')
args = parser.parse_args()

# ICESat biases: subtract this correction from elevation data

# Tim Urban - SLR [m]
BIAS_U = {20031027: -0.068 - 0.0,       # 2a   
          20040305: -0.052 - 0.0010635, # 2b 
          20040604: 0.006 - 0.0018165,  # 2c
          20041021: -0.038 - 0.0029505, # 3a
          20050307: -0.034 - 0.0040755, # 3b
          20050606: -0.001 - 0.0048345, # 3c 
          20051107: -0.003 - 0.0060915, # 3d
          20060311: -0.002 - 0.0071115, # 3e
          20060610: -0.009 - 0.007863,  # 3f
          20061111: 0.013 - 0.00912,    # 3g
          20070329: 0.008 - 0.010263,   # 3h
          20071019: -0.005 - 0.011934,  # 3i
          20080305: -0.014 - 0.0130635, # 3j
          20081012: -0.004 - 0.0148755, # 3k
          20081206: 0.016 - 0.01533,    # 2d
          20090326: 0.052 - 0.016239,   # 2e
          20091006: -0.011 - 0.017829}  # 2f

# Jay Zwally [m]
# a 0.0031 m/yr sea level correction has been applied.
BIAS_Z = {20031027: -0.0401, # 2a   
          20040305: 0.0508,  # 2b 
          20040604: 0.0361,  # 2c
          20041021: -0.0031, # 3a
          20050307: -0.0693, # 3b
          20050606: 0.0690,  # 3c 
          20051107: 0.0086,  # 3d
          20060311: -0.0014, # 3e
          20060610: -0.0312, # 3f
          20061111: 0.0715,  # 3g
          20070329: 0.0063,  # 3h
          20071019: 0.0386,  # 3i
          20080305: 0.0154,  # 3j
          20081012: -0.0685, # 3k
          20081206: -0.0829, # 2d
          20090326: 0,       # 2e
          20091006: 0}       # 2f

# Adrian Borsa (asc, des) [m]
BIAS_B = {20031027: (-0.0125, 0.025), # 2a   
          20040305: (0.051, 0.011),   # 2b 
          20040604: (0.0142, 0.057),  # 2c
          20041021: (0.011, 0.063),   # 3a
          20050307: (0.067, 0),       # 3b
          20050606: (0.051, 0.071),   # 3c 
          20051107: (0.076, 0.032),   # 3d
          20060311: (0.099, -0.019),  # 3e
          20060610: (-0.041, 0.093),  # 3f
          20061111: (0.017, -0.048),  # 3g
          20070329: (-0.048, 0.0120), # 3h
          20071019: (0.059, 0.042),   # 3i
          20080305: (0.087, 0.041),   # 3j
          20081012: (0, 0.0132),      # 3k
          20081206: (0.057, 0),       # 2d
          20090326: (0.093, -0.002),  # 2e
          20091006: (0, 0)}           # 2f


def wbias(nad, nda, a1, d1, a2, d2):
    '''
    Weighted average bias.
    nad, nda: 3D arrays
    a1, d1, a2, d2: 1D arrays
    '''
    a1 = a1[:,None,None]  # 1D -> 3D
    d1 = d1[:,None,None]
    a2 = a2[:,None,None]
    d2 = d2[:,None,None] 
    return (nad*(a2 - d1) + nda*(d2 - a1))/(nad + nda)


def get_biases_3d(t1, t2, nad, nda, source='urban'):
    if source == 'urban':
        b1 = np.asarray([BIAS_U[t] for t in t1])
        b2 = np.asarray([BIAS_U[t] for t in t2])
        bias = (b2 - b1)[:,None,None]  # 1D -> 3D
    elif source == 'zwally':
        b1 = np.asarray([BIAS_Z[t] for t in t1])
        b2 = np.asarray([BIAS_Z[t] for t in t2])
        bias = (b2 - b1)[:,None,None]  # 1D -> 3D
    elif source == 'borsa':
        b1 = np.asarray([BIAS_B[t] for t in t1])
        b2 = np.asarray([BIAS_B[t] for t in t2])
        bias = wbias(nad, nda, b1[:,0], b1[:,1], b2[:,0], b2[:,1]) # 3D
    else:
        ValueError('source must be urban/zwally/borsa')
    return bias


def get_data(fname_in, mode='r'):
    fin = tb.openFile(fname_in, mode)
    data = fin.root
    d = {}
    d['table'] = data.table
    d['time1'] = d['table'].cols.time1[:]
    d['time2'] = d['table'].cols.time2[:]
    d['dh_mean'] = data.dh_mean[:]
    d['n_ad'] = data.n_ad[:]
    d['n_da'] = data.n_da[:]
    return [d, fin]


def main(args):

    # input data -> dictionary and file
    fname_in = args.file[0]
    source = args.source
    d, fin = get_data(fname_in, 'a')
    print 'applying biases ...'

    # apply bias
    bias = get_biases_3d(d['time1'], d['time2'], d['n_ad'], d['n_da'], source=source)
    dh_bias = d[H_NAME] - bias

    # save
    if source == 'urban':
        SAVE_AS_NAME = 'dh_mean_u'
    elif source == 'zwally':
        SAVE_AS_NAME = 'dh_mean_z'
    elif source == 'borsa':
        SAVE_AS_NAME = 'dh_mean_b'
    else:
        raise
    shape =  dh_bias.shape    # i,j,k = t,y,x
    atom = tb.Atom.from_dtype(dh_bias.dtype)
    filters = tb.Filters(complib='zlib', complevel=9)
    c = fin.createCArray('/', SAVE_AS_NAME, atom, shape, '', filters)
    c[:] = dh_bias
    fin.close()
    print 'done.'
    print 'biases applied:', source, '->', SAVE_AS_NAME


if __name__ == '__main__':
    main(args)
