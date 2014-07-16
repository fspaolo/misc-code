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

try:
    sys.path.append('/Users/fpaolo/code/misc')
    from util import * 
    import util
    import viz
except:
    print("could'n import plotting functions!")
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
        d['satname_all'] = d['table_all'].cols.satname
        d['time1_all'] = d['table_all'].cols.time1
        d['time2_all'] = d['table_all'].cols.time2
        d['dh_mean_all'] = data.dh_mean_all
        d['dg_mean_all'] = data.dg_mean_all
    return [d, fin]


def backscatter_corr(H, G, term='mix', robust=False): 
    """
    Apply the backscatter correction to a dh time series.

    Implements the correction using a time series of dAGC formed
    exactly as the dh time series, following: Zwally, et al. (2005),
    Yi, et al. (2011):
        
        H_corr(t) = H(t) + S*G(t) + H0

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
    H_corr : corrected dh series
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
        return [H, None, None]

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
    #H_corr = H - S*(G - G0)
    H_corr = H - S*G + H0 

    return [H_corr, R, S]


def backscatter_corr2(H, G, term='mix', robust=False, npts=9):
    """
    Apply the backscatter correction to a dh time series.

    Implements the correction using a time series of dAGC formed
    exactly as the dh time series. It accounts for *time variable* 
    correlation, S(t).
        
        H_corr(t) = H(t) + S(t)*G(t) + H0(t)

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
    H_corr : corrected dh series
    R : correlation coeficient
    S : correlation gradient (sensitivity)

    Notes
    -----
    S is slope of linear fit to correlation(dG|dG', dH|dH')
    H0 is intercept of linear fit to correlation(dG|dG', dH|dH')
    RR, SS, HH are time series of the respective parameters.

    """
    if np.alltrue(np.isnan(H)):
        return [H, None, None]

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
    jj, = np.where(np.isnan(H))
    RR[jj] = np.nan
    SS[jj] = np.nan
    HH[jj] = np.nan

    H_corr = H - SS*G + HH
    H_corr = reference_to_first(H_corr)

    return [H_corr, RR, SS]


def reference_to_first(ts):
    ind, = np.where(~np.isnan(ts))
    if len(ind) > 1:
        ts -= ts[ind[0]]
    return ts 

#---------------------------------------------------------------------

def fy2ym(ftime2):
    """Decimal time2 -> time2, month."""
    fy, y  = np.modf(ftime2)
    m, y = int(np.ceil(fy*12)), int(y)
    return [y, m]


def y2dt(year):
    """
    Convert decimal year to `datetime` object.
    year : float or array_like of floats
    """
    if not np.iterable(year):
        year = np.asarray([year])
    ym = np.asarray([fy2ym(y) for y in year])
    dt = np.asarray([pn.datetime(y, m, 15) for y, m in ym])
    return dt


def i2dt(itimes):
    """
    Convert an integer representation of time YYYYMMDD to datetime.
    """
    return np.asarray([pn.datetime.strptime(str(it), '%Y%m%d') for it in itimes])


def plot_tseries(time2, lon, lat, dh_mean_corr, dh_mean, dg_mean, R, S, term='mix'):
    #time2 = y2dt(time2)
    R = np.mean(R)
    S = np.mean(S)
    if not np.alltrue(np.isnan(dh_mean[1:])):
        # use only non-null and non-zero entries for correlation
        ind, = np.where((~np.isnan(dh_mean)) & (~np.isnan(dg_mean)) & \
                        (dh_mean!=0) & (dg_mean!=0))
        t = np.arange(len(dh_mean))
        x, y = linear_fit(dg_mean[ind], dh_mean[ind])
        x2, y2 = linear_fit_robust(dg_mean[ind], dh_mean[ind])
        fig = plt.figure()
        ax = fig.add_subplot((111))
        plt.plot(dg_mean[ind], dh_mean[ind], 'o')
        plt.plot(x, y, linewidth=2, label='lstsq fit')
        plt.plot(x2, y2, linewidth=2, label='robust fit')
        plt.legend(loc=2).draw_frame(False)
        if term == 'mix':
            plt.xlabel('dAGC (dB)')
            plt.ylabel('dh (m)')
            plt.title('Mix-term sensitivity (S = corr grad)')
        elif term == 'short':
            plt.xlabel('$\Delta$dAGC (dB)')
            plt.ylabel('$\Delta$dh (m)')
            plt.title('Short-term sensitivity (S = corr grad)')
        ax = viz.add_inner_title(ax, 'R = %.2f,  S = %.2f' % (R, S), 4)
        #-----------------
        fig = plt.figure()
        plt.subplot((211))
        plt.plot(time2, dg_mean, linewidth=2, label='backscatter')
        plt.legend().draw_frame(False)
        plt.ylabel('dAGC (dB)')
        plt.title('lon = %.2f,  lat = %.2f' % (lon, lat))
        #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        plt.subplot((212))
        plt.plot(time2, dh_mean, 'g', linewidth=2, label='dh_uncorr')
        plt.plot(time2, dh_mean_corr, 'r', linewidth=2, label='dh_corr')
        plt.legend().draw_frame(False)
        plt.ylabel('dh (m)')
        #plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        fig.autofmt_xdate()
        return fig


def plot_map(lon, lat, RR, bbox, mfile, mres=1):
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = viz.make_proj_stere(bbox)
    m = viz.plot_grid_proj(m, lon, lat, RR, contourf=False, shift=True)
    plt.colorbar(orientation='vertical', shrink=.7)
    x, y, mask = viz.get_gtif_subreg(m, mfile, mres)
    mask = np.ma.masked_values(mask, 4)
    m.imshow(mask, zorder=1, cmap=plt.cm.gray_r)
    #m = plot_boundaries(m, (lon.min(),lon.max(),lat.min()-10,lat.max()+3))
    p_ = m.drawparallels(np.arange(-90.,-70.,3), labels=[0,0,1,0], color='0.6')
    m_ = m.drawmeridians(np.arange(-80.,-20.,5), labels=[1,0,0,0], color='0.6')
    return fig1 
