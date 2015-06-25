"""
Module containing functions and classes used by:

backscatter.py

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# February 7, 2012 

import os
import sys
import re
import numpy as np
import tables as tb
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import altimpy as ap

mpl.rcParams.update({'font.size': 16})

try:
    sys.path.append('/Users/fpaolo/code/misc')
    from util import * 
    import util
    import viz
except:
    print("could'n import plotting functions!")
    pass


class GetData(object):
    def __init__(self, fname, mode='r'):
        fin = tb.openFile(fname, mode)
        self.file = fin
        try:
            self.satname = fin.getNode('/satname')[:]
        except:
            self.satname = fname.split('_')[0].split('/')[-1]
        try:
            self.time1 = fin.getNode('/time1')[:]
            self.time2 = fin.getNode('/time2')[:]
        except:
            self.time = fin.getNode('/time')[:]
        self.lon = fin.getNode('/lon')[:]
        self.lat = fin.getNode('/lat')[:]
        self.x_edges = fin.getNode('/x_edges')[:]
        self.y_edges = fin.getNode('/y_edges')[:]
        self.dh_mean = fin.getNode('/dh_mean')[:]
        self.dh_error = fin.getNode('/dh_error')[:]
        self.dh_error2 = fin.getNode('/dh_error2')[:]
        self.dg_mean = fin.getNode('/dg_mean')[:]
        self.dg_error = fin.getNode('/dg_error')[:]
        self.dg_error2 = fin.getNode('/dg_error2')[:]
        self.n_ad = fin.getNode('/n_ad')[:]
        self.n_da = fin.getNode('/n_da')[:]
        try:
            self.time_all = fin.getNode('/time_all')[:]
            #self.dg_mean_all = fin.getNode('/dg_mean_all')[:]
            #self.dh_mean_all = fin.getNode('/dh_mean_all')[:]
            self.dg_mean_all_interp = fin.getNode('/dg_mean_all_interp')[:]
            self.dh_mean_all_interp = fin.getNode('/dh_mean_all_interp')[:]
            #self.dh_error_all = fin.getNode('/dh_error_all')[:]
            #self.dh_error2_all = fin.getNode('/dh_error2_all')[:]
            #self.dg_error_all = fin.getNode('/dg_error_all')[:]
            #self.dg_error2_all = fin.getNode('/dg_error2_all')[:]
        except:
            pass


def get_fname_out(fin, fname_out=None, sufix=None):
    """
    Construct the output file name with the min and max times 
    from the input file.
    """
    if fname_out is None:
        path, ext = os.path.splitext(fin)  # path from input file
        fname_out = '_'.join([path, sufix])
    return fname_out


def get_data_from_file(fname_in, node_name, mode='r', all_sat=False):
    d = {}
    fin = tb.openFile(fname_in, mode)
    data = fin.getNode('/', node_name)
    d['table'] = data.table
    d['satname'] = d['table'].cols.satname
    d['time1'] = d['table'].cols.time1
    d['time2'] = d['table'].cols.time2
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
    if all_sat:
        d['table_all'] = data.table_all
        d['satname'] = d['table_all'].cols.satname
        d['time1'] = d['table_all'].cols.time1
        d['time2'] = d['table_all'].cols.time2
        d['dh_mean_all'] = data.dh_mean_all
        d['dg_mean_all'] = data.dg_mean_all
    return [d, fin]


def plot_ts(time2, lon, lat, dh_mean_cor, dh_mean, dg_mean, R, S, diff=True):
    if np.alltrue(np.isnan(dh_mean[1:])):
        return None
    #time2 = y2dt(time2)
    R = np.mean(R)
    S = np.mean(S)
    # use only non-null and non-zero entries for correlation
    ind, = np.where((~np.isnan(dh_mean)) & (~np.isnan(dg_mean)) & \
                    (dh_mean!=0) & (dg_mean!=0))
    t = np.arange(len(dh_mean))

    if not diff:
        x, y = ap.linear_fit(dg_mean[ind], dh_mean[ind])
        x2, y2 = ap.linear_fit_robust(dg_mean[ind], dh_mean[ind])
        fig = plt.figure()
        ax = fig.add_subplot((111))
        plt.plot(dg_mean[ind], dh_mean[ind], 'o')
        plt.plot(x, y, linewidth=2, label='lstsq fit')
        plt.plot(x2, y2, linewidth=2, label='robust fit')
        plt.legend(loc=2).draw_frame(False)
        plt.xlabel('dAGC (dB)')
        plt.ylabel('dh (m)')
        plt.title('Mixed-term sensitivity')
    else:
        dh_mean2 = np.diff(dh_mean)
        dg_mean2 = np.diff(dg_mean)
        dh_mean2 = np.append(dh_mean2, np.nan)
        dg_mean2 = np.append(dg_mean2, np.nan)
        x, y = ap.linear_fit(dg_mean2[ind], dh_mean2[ind])
        x2, y2 = ap.linear_fit_robust(dg_mean2[ind], dh_mean2[ind])
        fig = plt.figure()
        ax = fig.add_subplot((111))
        plt.plot(dg_mean2[ind], dh_mean2[ind], 'o')
        plt.plot(x, y, linewidth=2, label='lstsq fit')
        plt.plot(x2, y2, linewidth=2, label='robust fit')
        plt.legend(loc=2).draw_frame(False)
        plt.xlabel('$\Delta$dAGC (dB)')
        plt.ylabel('$\Delta$dh (m)')
        plt.title('Short-term sensitivity')
    ax1 = viz.add_inner_title(ax, 'corrcoef: R = %.2f' % R, 3)
    ax1 = viz.add_inner_title(ax, 'slope: S = %.2f' % S, 4)
    plt.savefig('corr.png')
    #-----------------
    if not diff:
        fig = plt.figure()
        ax2 = plt.subplot((211))
        plt.plot(time2, dh_mean, 'b', linewidth=2, label='dh')
        plt.plot(time2, dh_mean_cor, 'r', linewidth=2, label='dh$_{COR}$')
        #plt.legend().draw_frame(False)
        viz.add_inner_title(ax2, 'dh', 2)
        viz.add_inner_title(ax2, 'dh$_{COR}$', 3)
        plt.title('lon = %.2f,  lat = %.2f' % (lon, lat))
        plt.ylabel('m')
        #plt.xlim(1992, 2012.1)
        #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax3 = plt.subplot((212))
        plt.plot(time2, dg_mean, 'g', linewidth=2, label='dAGC')
        #plt.legend().draw_frame(False)
        viz.add_inner_title(ax3, 'dAGC', 3)
        plt.ylabel('dB')
        #plt.xlim(1992, 2012.1)
        #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    else:
        fig = plt.figure()
        ax2 = plt.subplot((311))
        plt.plot(time2, dh_mean, 'b', linewidth=2, label='dh')
        plt.plot(time2, dh_mean_cor, 'r', linewidth=2, label='dh$_{COR}$')
        #plt.legend().draw_frame(False)
        viz.add_inner_title(ax2, 'dh', 2)
        viz.add_inner_title(ax2, 'dh$_{COR}$', 3)
        plt.title('lon = %.2f,  lat = %.2f' % (lon, lat))
        plt.ylabel('m')
        #plt.xlim(1992, 2012.1)
        #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax3 = plt.subplot((312))
        plt.plot(time2, dh_mean2, 'm', linewidth=2, label='$\Delta$dh')
        #plt.legend().draw_frame(False)
        viz.add_inner_title(ax3, '$\Delta$dh', 3)
        plt.ylabel('m')
        #plt.xlim(1992, 2012.1)
        #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax4 = plt.subplot((313))
        plt.plot(time2, dg_mean2, 'c', linewidth=2, label='$\Delta$dAGC')
        #plt.legend().draw_frame(False)
        viz.add_inner_title(ax4, '$\Delta$dAGC', 3)
        plt.ylabel('dB')
        #plt.xlim(1992, 2012.1)
        #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    fig.autofmt_xdate()
    plt.savefig('ts.png')
    return fig

def plot_rs(time2, R, S):
    if np.alltrue(np.isnan(R)):
        return None
    fig = plt.figure(figsize=(9,2.5))
    ax1 = fig.add_subplot((111))
    ax2 = plt.twinx()
    p1, = ax1.plot(time2, R, 'b', linewidth=3, label='correlation, $\\rho(t)$')
    p2, = ax2.plot(time2, S, 'r', linewidth=3, label='sensitivity, $s(t)$')
    ax1.yaxis.label.set_color(p1.get_color())
    ax2.yaxis.label.set_color(p2.get_color())
    ax1.set_ylabel('Correlation')
    ax2.set_ylabel('Sensitivity')
    #ax1.set_ylim(0.4, 1.0)
    #ax2.set_ylim(0.0, 0.5)
    [(l.set_rotation(45), l.set_fontsize(16)) for l in ax1.xaxis.get_ticklabels()]
    lines = [p1, p2]
    fig.autofmt_xdate()
    return fig


def plot_map(lon, lat, grid, bbox, mfile, mres=1, **kw):
    """
    **kw : keyword args
        contourf=[True/False]
        vmin=int
        vmax=int
    """
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = viz.make_proj_stere(bbox)
    m = viz.plot_grid_proj(m, lon, lat, grid, shift=True, **kw)
    plt.colorbar(orientation='vertical', shrink=.65)
    x, y, mask = viz.get_gtif_subreg(m, mfile, mres)
    mask = np.ma.masked_values(mask, 4)
    m.imshow(mask, zorder=1, cmap=plt.cm.gray_r)
    #m = plot_boundaries(m, (lon.min(),lon.max(),lat.min()-10,lat.max()+3))
    p_ = m.drawparallels(np.arange(-90.,-60,3), labels=[0,0,0,1], color='0.2')
    m_ = m.drawmeridians(np.arange(-180,180.,5), labels=[1,0,0,0], color='0.2')
    return fig1 
