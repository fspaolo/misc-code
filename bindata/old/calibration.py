#!/usr/bin/env python

"""
$ pwd
/Users/fpaolo/code/bindata

$ python calibration.py ~/data/fris/xover/seasonal/tseries_mean.h5
"""

import os
import sys
import re
import numpy as np
import tables as tb
import pandas as pn
import matplotlib.pyplot as plt
from dateutil import parser

sys.path.append('/Users/fpaolo/code')
from filter import filter_ts as filt

def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size, pad=0., 
        borderpad=0.5, frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


def main(argv):
    db = tb.openFile(argv[1])
    ts = db.root.ts_mean
    ts_ers1 = ts.readWhere('(sat == "ers1") & (dh != 0)')
    ts_ers2 = ts.readWhere('sat == "ers2"')
    ts_envisat = ts.readWhere('sat == "envisat"')
    db.close()

    # inter-satellite bias
    bias = np.mean(ts_ers1['dh'][-5:] - ts_ers2['dh'][:5])
    ts_ers2['dh'] += bias
    bias = np.mean(ts_ers2['dh'][-4:] - ts_envisat['dh'][:4])
    ts_envisat['dh'] += bias

    t1 = [parser.parse(d) for d in ts_ers1['time']]
    t2 = [parser.parse(d) for d in ts_ers2['time']]
    t3 = [parser.parse(d) for d in ts_envisat['time']]
    t_all = np.concatenate((t1, t2, t3))
    dh_all = np.concatenate((ts_ers1['dh'], ts_ers2['dh'], ts_envisat['dh']))
    se_all = np.concatenate((ts_ers1['se_dh'], ts_ers2['se_dh'], ts_envisat['se_dh']))
    t_all, ind = np.unique(t_all, return_index=True)
    dh_all = dh_all[ind]
    se_all = se_all[ind]

    ts_ers2['dh'][0] = np.nan     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ts_ers2['dh'][1] = np.nan     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ts_ers2['dh'][2] -= 0.3     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ts_ers2['dh'][3] += 0.38     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ts_ers2['dh'] += 0.4     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ts_envisat['dh'] -= 0.05  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #-----------------------------------------------------------------

    fig = plt.figure()
    ax1 = fig.add_subplot(211, axisbg='#FFF8DC')
    plt.ylim(-1, 1.2)
    plt.ylabel('Elevation change (m)')
    ax2 = fig.add_subplot(212, axisbg='#FFF8DC')
    plt.ylim(-1.01, .75)
    plt.ylabel('Elevation change (m)')

    colors = ('b', 'g', 'r')
    sats = (ts_ers1, ts_ers2, ts_envisat)
    times = (t1, t2, t3)
    for s, t, c in zip(sats, times, colors):
        dh = s['dh']
        se = s['se_dh']
        #ax1.errorbar(t, dh, yerr=se, fmt='.', color=c, linewidth=1)
        ax1.plot(t, dh, color=c, linewidth=2)
        ax1.fill_between(t, dh+se, dh-se, color=c, alpha=0.4)

    sat = ('ERS-1', 'ERS-2', 'Envisat')
    loc = (2, 9, 1)
    for s, l in zip(sat, loc):
        t = add_inner_title(ax1, s, loc=l)
        t.patch.set_alpha(0.5)
    t = add_inner_title(ax1, 'Need cross-calibration (single grid cell)', loc=3)

    #-----------------------------------------------------------------

    # filter
    #flt = filt.FilterData(dh_all)
    #dh_all_f = flt.time_filt(dh_all, window_len=5)

    # fit trend
    def poly_fit(t_all, dh_all, n=3):
        t_all_n = [pn.datetime.toordinal(i) for i in t_all]
        coeff = np.polyfit(t_all_n, dh_all, n)
        t_fit_n = np.arange(t_all_n[0], t_all_n[-1], 20)
        yfit = np.polyval(coeff, t_fit_n)
        t_fit = [pn.datetime.fromordinal(i) for i in t_fit_n]
        return t_fit, yfit, coeff[0]

    #-----------------------------------------------------------------

    def spline_fit(t_all, dh_all, s=0.2, k=3):
        t_all_n = [pn.datetime.toordinal(i) for i in t_all]
        t_fit_n = np.arange(t_all_n[0], t_all_n[-1], 20)
        t_fit = [pn.datetime.fromordinal(i) for i in t_fit_n]

        from scipy.interpolate import splprep, splev

        # spline parameters
        #s = 0.2 # smoothness parameter
        #k = 3 # spline order
        nest = -1 # estimate of number of knots needed (-1 = maximal)

        # find the knot points
        tckp, u = splprep([dh_all, t_all_n], s=s, k=k, nest=-1)

        # evaluate spline, including interpolated points
        dh_fit, t_tmp = splev(np.linspace(0, 1, len(t_fit_n)), tckp)
        return t_tmp, dh_fit

    #-----------------------------------------------------------------

    t_fit, dh_fit = spline_fit(t_all, dh_all, k=3)
    t_fit, yfit, coeff = poly_fit(t_all, dh_all, n=20)
    t_fit, yfit2, coeff = poly_fit(t_all, dh_all, n=1)
    
    ax2.plot(t_fit, dh_fit, 'b', linewidth=3)
    ax2.plot(t_all, dh_all, 'xb')
    #ax2.errorbar(t_all, dh_all, yerr=se_all, color='b', marker='x')
    ax2.plot(t_fit, yfit, 'r', linewidth=2)
    ax2.plot(t_fit, yfit2, '--k', linewidth=2)

    rate = ((yfit2[-1] - yfit2[0]) / 17.) * 100   #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    t = add_inner_title(ax2, 'Linear trend = %.1f cm/yr (single grid cell)' % rate, loc=3)
    t.patch.set_alpha(0.5)

    fig.autofmt_xdate()
    plt.savefig('ts_final.pdf', dpi=150, bbox_inches='tight')
    os.system('cp ts_final.pdf ~/posters/agu2011/figures/')
    plt.show()


if __name__ == '__main__':
    main(sys.argv)

