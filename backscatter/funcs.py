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
import pandas as pn
import matplotlib.pyplot as plt

sys.path.append('/Users/fpaolo/code/misc')
from util import * 
from viz import plot_matrix, add_inner_title


def get_fname_out(file_in, fname_out=None, sufix=None):
    """
    Construct the output file name with the min and max times 
    from the input file.
    """
    if fname_out is None:
        path, ext = os.path.splitext(file_in)  # path from input file
        fname_out = '_'.join([path, sufix])
    return fname_out


def get_data(fname_in, node_name, mode='r'):
    d = {}
    file_in = tb.openFile(fname_in, mode)
    data = file_in.getNode('/', node_name)
    table = data.table
    d['sat_name'] = table.cols.sat_name
    d['ref_time'] = table.cols.ref_time
    d['year'] = table.cols.year
    d['month'] = table.cols.month
    d['dh_mean'] = data.dh_mean
    d['dh_error'] = data.dh_error
    d['dg_mean'] = data.dg_mean
    d['dg_error'] = data.dg_error
    d['n_ad'] = data.n_ad
    d['n_da'] = data.n_da
    d['x_edges'] = data.x_edges
    d['y_edges'] = data.y_edges
    d['lon'] = data.lon
    d['lat'] = data.lat
    return d, file_in


def backscatter_corr(H, G, robust=False): 
    """
    Apply the backscatter correction to a dh time series.

    Implements the correction using a time series of dAGC following
    Zwally, et al. (2005).

    Parameters
    ----------
    H : time series of dh (m)
    G : time series of dAGC (dB)
    robust : performs linear fit by robust regression (M-estimate),
        otherwise uses Ordinary Least Squares (default).

    Returns
    -------
    H_corr : corrected dh series
    R : correlation coeficient
    dHdG : sensitivity factor

    Notes
    -----
    dHdG is slope of linear fit to correlation(dAGC,dh)
    H0 is intercept of linear fit to correlation(dAGC,dh)
    """
    # use only valid entries
    ind, = np.where((~np.isnan(H)) & (~np.isnan(G)))
    if len(ind) < 2: return H, 0, 0
    H2, G2 = H[ind], G[ind]

    # correlation coef
    R = np.corrcoef(G2, H2)[0,1]
    # sesitivity factor
    if robust:
        dHdG, H0 = linear_fit_robust(G2, H2, return_coef=True)
    else:
        dHdG, H0 = linear_fit(G2, H2, return_coef=True)

    # no correction applied if |R| < 0.2
    if np.abs(R) < 0.2:                          
        return H, R, dHdG
    elif dHdG < -0.2:
        dHdG = -0.2
    elif dHdG > 0.7:
        dHdG = 0.7
    G0 = -(1./dHdG)*H0
    H_corr = H - dHdG*(G - G0)
    return H_corr, R, dHdG


def plot_figures(dh_mean_corr, dh_mean, dg_mean, R, dHdG):
    if not np.alltrue(np.isnan(dh_mean[1:])):
        t = np.arange(len(dh_mean))
        tt, dg_mean_interp = spline_interp(t, dg_mean)
        x, y = linear_fit(dg_mean, dh_mean)
        x2, y2 = linear_fit_robust(dg_mean, dh_mean)
        fig = plt.figure()
        plt.plot(dg_mean, dh_mean, 'o')
        plt.plot(x, y, linewidth=2)
        plt.plot(x2, y2, linewidth=2)
        fig = plt.figure()
        ax = fig.add_subplot((211))
        plt.plot(tt, dg_mean_interp, linewidth=2)
        ax = fig.add_subplot((212))
        ax = add_inner_title(ax, 'R = %.2f, dH/dG = %.2f' % (R, dHdG), 2)
        plt.plot(t, dh_mean, linewidth=2)
        plt.plot(t, dh_mean_corr, linewidth=2)
        plt.show()
