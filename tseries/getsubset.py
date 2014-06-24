#!/usr/bin/env python

import os
import sys
import numpy as np
import tables as tb
import argparse as ap
import matplotlib.pyplot as plt

#from funcs import *

SAVE_TO_FILE = True
NODE_NAME = ''

class TimeSeriesGrid(tb.IsDescription):
    satname = tb.StringCol(20, pos=1)
    time1 = tb.Int32Col(pos=2)
    time2 = tb.Int32Col(pos=3)

def get_data_from_file(fname_in, node_name, mode='r'):
    d = {}
    fin = tb.openFile(fname_in, mode)
    data = fin.getNode('/', node_name)
    d['table'] = data.table
    d['sat_name'] = d['table'].cols.sat_name
    d['ref_time'] = d['table'].cols.ref_time
    d['year'] = d['table'].cols.year
    d['lon'] = data.lon
    d['lat'] = data.lat
    d['x_edges'] = data.x_edges
    d['y_edges'] = data.y_edges
    d['n_ad'] = data.n_ad
    d['n_da'] = data.n_da
    d['dh_mean'] = data.dh_mean
    d['dh_error'] = data.dh_error
    d['dh_error2'] = data.dh_error2
    d['dg_mean'] = data.dg_mean
    d['dg_error'] = data.dg_error
    d['dg_error2'] = data.dg_error2
    ###
    d['table_all'] = data.table_all
    d['sat_name_all'] = d['table_all'].cols.sat_name
    d['ref_time_all'] = d['table_all'].cols.ref_time
    d['year_all'] = d['table_all'].cols.year
    d['dh_mean_all'] = data.dh_mean_all
    d['dg_mean_all'] = data.dg_mean_all
    return [d, fin]


def year2ym(fyear):
    """Decimal year -> year, month."""
    fy, y  = np.modf(fyear)
    m, y = int(np.ceil(fy*12)), int(y)
    if (m == 0): m = 1
    return [y, m]

def year2int(year, day=15):
    if not np.iterable(year):
        year = np.asarray([year])
    ym = [year2ym(y) for y in year]
    return [int(y*10000 + m*100 + day) for y,m in ym]

def main():

    fname_in = sys.argv[1]
    path, _ = os.path.splitext(fname_in)
    fname_out = path + '_3m.h5'

    d, fin = get_data_from_file(fname_in, NODE_NAME, 'r')

    dg_error = d['dg_error'][:]
    dg_error2 = d['dg_error2'][:]
    dg_mean = d['dg_mean'][:]
    dg_mean_all = d['dg_mean_all'][:]
    dh_error = d['dh_error'][:]
    dh_error2 = d['dh_error2'][:]
    dh_mean = d['dh_mean'][:]
    dh_mean_all = d['dh_mean_all'][:]
    lat = d['lat'][:]
    lon = d['lon'][:]
    n_ad = d['n_ad'][:]
    n_da = d['n_da'][:]
    table = d['table']
    sat_name = d['sat_name'][:]
    ref_time = d['ref_time'][:]
    year = d['year'][:]
    table_all = d['table_all']
    sat_name_all = d['sat_name_all'][:]
    ref_time_all = d['ref_time_all'][:]
    year_all = d['year_all'][:]
    y_edges = d['y_edges'][:]
    x_edges = d['x_edges'][:]

    # extract region
    i, = np.where((-82 <= lat) & (lat <= -79))
    j, = np.where((169.5 <= lon) & (lon <= 180.5))

    lon = lon[j]
    lat = lat[i]
    dg_error = dg_error[:,i,:]
    dg_error = dg_error[:,:,j]
    dg_error2 = dg_error2[:,i,:]
    dg_error2 = dg_error2[:,:,j]
    dg_mean = dg_mean[:,i,:]
    dg_mean = dg_mean[:,:,j]
    dg_mean_all = dg_mean_all[:,i,:]
    dg_mean_all = dg_mean_all[:,:,j]
    dh_error = dh_error[:,i,:]
    dh_error = dh_error[:,:,j]
    dh_error2 = dh_error2[:,i,:]
    dh_error2 = dh_error2[:,:,j]
    dh_mean = dh_mean[:,i,:]
    dh_mean = dh_mean[:,:,j]
    dh_mean_all = dh_mean_all[:,i,:]
    dh_mean_all = dh_mean_all[:,:,j]
    n_ad = n_ad[:,i,:]
    n_ad = n_ad[:,:,j]
    n_da = n_da[:,i,:]
    n_da = n_da[:,:,j]

    i, = np.where((-82 <= y_edges) & (y_edges <= -72))
    j, = np.where((170 <= x_edges) & (x_edges <= 180))

    y_edges = y_edges[i]
    x_edges = x_edges[j]

    # fill with NaNs at the borders
    dg_error[:,:,0] = np.nan
    dg_error[:,:,-1] = np.nan
    dg_error2[:,:,0] = np.nan
    dg_error2[:,:,-1] = np.nan
    dg_mean[:,:,0] = np.nan
    dg_mean[:,:,-1] = np.nan
    dg_mean_all[:,:,0] = np.nan
    dg_mean_all[:,:,-1] = np.nan
    dh_error[:,:,0] = np.nan
    dh_error[:,:,-1] = np.nan
    dh_error2[:,:,0] = np.nan
    dh_error2[:,:,-1] = np.nan
    dh_mean[:,:,0] = np.nan
    dh_mean[:,:,-1] = np.nan
    dh_mean_all[:,:,0] = np.nan
    dh_mean_all[:,:,-1] = np.nan

    # convert time representation
    time2 = year2int(year)
    time2_all = year2int(year_all)
    time1 = np.empty(len(time2), 'i4')
    time1_all = np.empty(len(time2_all), 'i4')
    time1.fill(time2[0])
    time1_all.fill(time2_all[0])

    '''
    plt.imshow(dh_mean[10], interpolation='nearest')
    plt.show()
    sys.exit()
    '''

    #---------------------------------------------------------

    # save to file
    filters = tb.Filters(complib='zlib', complevel=9)
    fout = tb.openFile(fname_out, 'w')
    fout.createArray('/', 'dg_error', dg_error)
    fout.createArray('/', 'dg_error2', dg_error2)
    fout.createArray('/', 'dg_mean', dg_mean)
    fout.createArray('/', 'dg_mean_all', dg_mean_all)
    fout.createArray('/', 'dh_error', dh_error)
    fout.createArray('/', 'dh_error2', dh_error2)
    fout.createArray('/', 'dh_mean', dh_mean)
    fout.createArray('/', 'dh_mean_all', dh_mean_all)
    fout.createArray('/', 'lon', lon)
    fout.createArray('/', 'lat', lat)
    fout.createArray('/', 'x_edges', x_edges)
    fout.createArray('/', 'y_edges', y_edges)
    fout.createArray('/', 'n_ad', n_ad)
    fout.createArray('/', 'n_da', n_da)
    t1 = fout.createTable('/', 'table', TimeSeriesGrid, '', filters)
    t2 = fout.createTable('/', 'table_all', TimeSeriesGrid, '', filters)
    t1.append((sat_name, time1, time2)) 
    t2.append((sat_name_all, time1_all, time2_all)) 
    t1.flush()
    t2.flush()

    fout.close()
    fin.close()

    if SAVE_TO_FILE:
        print 'out file -->', fname_out


if __name__ == '__main__':
    main()
