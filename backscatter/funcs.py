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


def backscatter_corr(H, G, term='mix', robust=False): 
    """
    Apply the backscatter correction to a dh time series.

    Implements the correction using a time series of dAGC formed
    exactly as the dh time series, following: Zwally, et al. (2005),
    Yi, et al. (2011):
        
        H_cor(t) = H(t) - S*G(t) - H0

    where H(t) = dh(t0,t), G(t) = dAGC(t0,t) and S = dH/dG = const.

    Parameters
    ----------
    H : time series of dh (m)
    G : time series of dAGC (dB)
    term : if `mix`, derive mix-term sensitivity using dH/dG; 
        if `short`, derive short-term sensitivity using the derivatives dH'/dG'.
    robust : performs linear fit by robust regression (M-estimate),
        otherwise uses Ordinary Least Squares (default).

    Returns
    -------
    H_cor : corrected dh series
    R : correlation coeficient
    S : sensitivity factor

    Notes
    -----
    S is slope of linear fit to correlation(dG|dG', dH|dH')
    H0 is intercept of linear fit to correlation(dG|dG', dH|dH')

    """
    # use only non-null and non-zero entries for correlation
    ind, = np.where((~np.isnan(H)) & (~np.isnan(G)) & (H!=0) & (G!=0))
    if len(ind) < 2: 
        return [H, np.nan, np.nan]

    if term == 'mix':
        H2, G2 = H[ind], G[ind]
    elif term == 'short':
        H2, G2 = np.diff(H[ind]), np.diff(G[ind])
    else:
        raise IOError('`term` must be "mix" or "short"')

    # correlation coef
    R = np.corrcoef(G2, H2)[0,1]

    # correlation grad and intercept
    if robust:
        S, H0 = linear_fit_robust(G2, H2, return_coef=True)
    else:
        S, H0 = linear_fit(G2, H2, return_coef=True)

    # no correction applied if |R| < 0.2
    if np.abs(R) < 0.2:                          
        return [H, R, S]
    elif S < -0.2:
        S = -0.2
    elif S > 0.7:
        S = 0.7
    #G0 = -H0*(1./S)
    #H_cor = H - S*(G - G0)
    H_cor = H - S*G - H0 

    return [H_cor, R, S]


def backscatter_corr2(H, G, term='mix', robust=False, npts=9):
    """
    Apply the backscatter correction to a dh time series.

    Implements the correction using a time series of dAGC formed
    exactly as the dh time series. It accounts for *time variable* 
    correlation, S(t).
        
        H_cor(t) = H(t) - S(t)*G(t) - H0(t)

    where H(t) = dh(t0,t), G(t) = dAGC(t0,t) and S(t) = dH/dG(t).
 
    Parameters
    ----------
    H : time series of dh (m)
    G : time series of dAGC (dB)
    term : if `mix`, derive mix-term sensitivity using dH/dG; 
        if `short`, derive short-term sensitivity using differences DdH/DdG.
    robust : performs linear fit by robust regression (M-estimate),
        otherwise uses Ordinary Least Squares (default).
    npts : number of points used for correlation at each time.

    Returns
    -------
    H_cor : corrected dh series
    R : correlation coeficient
    S : correlation gradient (sensitivity)

    Notes
    -----
    S is slope of linear fit to correlation(dG|dG', dH|dH')
    H0 is intercept of linear fit to correlation(dG|dG', dH|dH')
    RR, SS, HH are time series of the respective parameters.

    """
    if np.alltrue(np.isnan(H)):
        return [H, np.nan, np.nan]

    H = reference_to_first(H)
    G = reference_to_first(G)

    N = len(H)
    RR = np.empty(N, 'f8') * np.nan
    SS = np.empty(N, 'f8') * np.nan
    HH = np.empty(N, 'f8') * np.nan
    l = int(npts/2.)

    for k in range(N):
        if k < l or k >= N-l: 
            continue
        H2, G2 = H[k-l:k+l+1], G[k-l:k+l+1]    # take chunks every iteration
        if term == 'mix':
            pass
        elif term == 'short':
            H2, G2 = np.diff(H2), np.diff(G2)  # differences
        else:
            raise IOError('`term` must be "mix" or "short"')

        # correlation coef
        R = np.corrcoef(G2, H2)[0,1]

        # correlation grad and intercept
        if robust:
            S, H0 = linear_fit_robust(G2, H2, return_coef=True)
        else:
            S, H0 = linear_fit(G2, H2, return_coef=True)

        RR[k] = R
        SS[k] = S
        HH[k] = H0 

    # fill both ends
    RR[:l] = RR[l]
    SS[:l] = SS[l]
    HH[:l] = HH[l]
    RR[N-l:] = RR[N-l-1]
    SS[N-l:] = SS[N-l-1]
    HH[N-l:] = HH[N-l-1]

    # no correction applied if |R| < 0.2
    ii, = np.where(np.abs(RR) < 0.2)
    SS[ii] = 0.0
    HH[ii] = 0.0
    SS[SS<-0.2] = -0.2
    SS[SS>0.7] = 0.7

    # fill with NaN when no data in dh TS
    jj, = np.where(np.isnan(H))
    RR[jj] = np.nan
    SS[jj] = np.nan
    HH[jj] = np.nan

    H_cor = H - SS*G - HH
    H_cor = reference_to_first(H_cor)

    return [H_cor, RR, SS]


def backscatter_corr3(H, G, t, intervals, term='mix', robust=False):
    """
    Apply the backscatter correction to a dh time series.

    Implements the correction using a time series of dAGC formed
    exactly as the dh time series. It calculates the correlation
    and regression slope (sensitivity) for given time intervals:
        
        dh_cor(t) = dh(t) - s(intv)*dg(t) - h0(intv)

    where dh(t) = dh(t0,t), dg(t) = dAGC(t0,t) and s(intv) = dh/dg[intv].
 
    Parameters
    ----------
    H : time series of dh (m)
    G : time series of dAGC (dB)
    t : times in datetime objects 
    intervals : list of `datetime` tuples defining the time 
        intervals: [(dt1,dt2), (dt2,dt3),...]
    term : if `mix`, derive mix-term sensitivity using dH/dG; 
        if `short`, derive short-term sensitivity using differences DdH/DdG.
    robust : performs linear fit by robust regression (M-estimate),
        otherwise uses Ordinary Least Squares (default).

    Returns
    -------
    H_cor : corrected dh series
    R : correlation coeficient
    S : correlation gradient (sensitivity)

    Notes
    -----
    S is slope of linear fit to correlation(dG|dG', dH|dH')
    H0 is intercept of linear fit to correlation(dG|dG', dH|dH')
    RR, SS, HH are time series of the respective parameters.

    """
    if np.alltrue(np.isnan(H)):
        return [H, np.nan, np.nan]

    H = reference_to_first(H)
    G = reference_to_first(G)

    N = len(H)
    RR = np.empty(N, 'f8') * np.nan
    SS = np.empty(N, 'f8') * np.nan
    HH = np.empty(N, 'f8') * np.nan

    for tt in intervals:
        t1, t2 = tt
        kk, = np.where((t >= t1) & (t < t2))
        # take chunks (intervals) every iteration
        H2, G2 = H[kk], G[kk]    
        if term == 'mix':
            pass
        elif term == 'short':
            H2, G2 = np.diff(H2), np.diff(G2)  # differences
        else:
            raise IOError('`term` must be "mix" or "short"')

        # correlation coef
        R = np.corrcoef(G2, H2)[0,1]

        # correlation grad and intercept
        if robust:
            S, H0 = linear_fit_robust(G2, H2, return_coef=True)
        else:
            S, H0 = linear_fit(G2, H2, return_coef=True)

        RR[kk] = R
        SS[kk] = S
        HH[kk] = H0 

    # no correction applied if |R| < 0.2
    ii, = np.where(np.abs(RR) < 0.2)
    SS[ii] = 0.0
    HH[ii] = 0.0
    SS[SS<-0.2] = -0.2
    SS[SS>0.7] = 0.7

    # fill with NaN when no data in dh TS
    jj, = np.where(np.isnan(H))
    RR[jj] = np.nan
    SS[jj] = np.nan
    HH[jj] = np.nan

    H_cor = H - SS*G - HH
    H_cor = reference_to_first(H_cor)

    return [H_cor, RR, SS]


def reference_to_first(ts):
    ind, = np.where(~np.isnan(ts))
    if len(ind) > 1:
        ts -= ts[ind[0]]
    return ts 

#---------------------------------------------------------------------

def year2ym(fyear):
    """Decimal year -> year, month."""
    fy, y  = np.modf(fyear)
    m, y = int(np.ceil(fy*12)), int(y)
    if (m == 0): m = 1
    return [y, m]


def year2dt(year):
    """
    Convert decimal year to `datetime` object.
    year : float [array], decimal years.
    """
    if not np.iterable(year):
        year = np.asarray([year])
    ym = np.asarray([year2ym(y) for y in year])
    dt = np.asarray([pd.datetime(y, m, 15) for y, m in ym])
    return dt


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
        x, y = linear_fit(dg_mean[ind], dh_mean[ind])
        x2, y2 = linear_fit_robust(dg_mean[ind], dh_mean[ind])
        fig = plt.figure()
        ax = fig.add_subplot((111))
        plt.plot(dg_mean[ind], dh_mean[ind], 'o')
        plt.plot(x, y, linewidth=2, label='lstsq fit')
        plt.plot(x2, y2, linewidth=2, label='robust fit')
        plt.legend(loc=2).draw_frame(False)
        plt.xlabel('dAGC (dB)')
        plt.ylabel('dh (m)')
        plt.title('Mix-term sensitivity')
    else:
        dh_mean2 = np.diff(dh_mean)
        dg_mean2 = np.diff(dg_mean)
        dh_mean2 = np.append(dh_mean2, np.nan)
        dg_mean2 = np.append(dg_mean2, np.nan)
        x, y = linear_fit(dg_mean2[ind], dh_mean2[ind])
        x2, y2 = linear_fit_robust(dg_mean2[ind], dh_mean2[ind])
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
