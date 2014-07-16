import sys
import numpy as np
import tables as tb
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import altimpy as ap

fname = '/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5'

def gradient(y, dt=.25):
    return np.gradient(y.values, dt)

with tb.openFile(fname) as f:

    alt0 = f.root.dh_mean_xcal[:]
    #alt0 = f.root.dh_mean_xcal_short_const[:]
    alt1 = f.root.dg_mean_xcal[:]
    firn = f.root.h_firn[:]
    t_alt = ap.num2date(f.root.time_xcal[:])
    t_firn = ap.num2date(f.root.time_firn[:])
    lon = f.root.lon[:]
    lat = f.root.lat[:]

    # map grid indexes to coordinates
    ii, jj = xrange(len(lat)), xrange(len(lon))
    ij2ll = dict([((i,j),(la,lo)) for i,j,la,lo in zip(ii,jj,lat,lon)])

    df_alt0 = pd.Panel(alt0, items=t_alt, major_axis=ii, minor_axis=jj
                      ).to_frame(filter_observations=False).T
    df_alt1 = pd.Panel(alt1, items=t_alt, major_axis=ii, minor_axis=jj
                      ).to_frame(filter_observations=False).T
    df_firn = pd.Panel(firn, items=t_firn, major_axis=ii, minor_axis=jj
                      ).to_frame().T

    # 3-month averages, and referenced to first
    df_firn = df_firn.resample('Q-NOV', loffset='-45D', axis=0).apply(ap.referenced, to='mean')

    # filter-out time series
    df_alt0 = df_alt0[df_firn.columns].drop(df_alt0.index[-1])
    df_alt1 = df_alt1[df_firn.columns].drop(df_alt1.index[-1])
    df_alt1 = df_alt1.where(df_alt1.abs() < 10).dropna(axis=1)
    df_alt0 = df_alt0[df_alt1.columns]
    df_firn = df_firn[df_alt1.columns]
    df_alt0.index = df_alt1.index = df_firn.index

    df_alt0 = df_alt0.apply(ap.referenced, to='mean')
    df_alt1 = df_alt1.apply(ap.referenced, to='mean')

    df_alt0 = df_alt0.apply(gradient, dt=.25)
    df_alt1 = df_alt1.apply(gradient, dt=.25)
    df_firn = df_firn.apply(gradient, dt=.25)

    #------------------------- FIGURES --------------------------

    ap.rcparams()

    # time series
    '''
    ax = df_alt1.plot(legend=False)
    plt.title('Individual time series, all ice shelves')
    ap.intitle('Altimetry', 2, ax=ax)
    plt.ylabel('Surface elevation (m)')
    plt.savefig('tseries_altim_all.png', bbox_inches='tight')
    ax2 = df_firn.plot(legend=False)
    plt.title('Individual time series, all ice shelves')
    ap.intitle('Firn model', 2, ax=ax2)
    plt.ylabel('Surface elevation (m)')
    plt.savefig('tseries_firn_all.png', bbox_inches='tight')
    plt.show()
    '''
    ###
    '''
    plt.figure()
    #df_alt0.mean(axis=1).plot(linewidth=2, label='uncorr_altim')
    #df_alt1.mean(axis=1).plot(linewidth=2, label='corr_altim')
    df_alt1.mean(axis=1).plot(linewidth=2, label='backscatter')
    df_firn.mean(axis=1).plot(linewidth=2, label='firn_model', color='r')
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02)).draw_frame(False)
    plt.title('Average time series, all ice shelves')
    #plt.ylabel('Surface elevation (m)')
    plt.ylabel('meters or dB')
    plt.savefig('tseries_bs.png', bbox_inches='tight')
    plt.show()
    '''

    # correlation of time series
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(121)
    df_alt = df_alt0.copy()
    alt, firn = df_alt0.unstack().values, df_firn.unstack().values
    i, = np.where((~np.isnan(alt)) & (~np.isnan(firn)))
    corrcoef = np.corrcoef(alt[i], firn[i])[0,1]
    plt.scatter(df_alt.unstack(), df_firn.unstack(), marker='.', color='b', alpha=.5)
    plt.title('Individual time series')
    ap.intitle('correlation = %.2f' % corrcoef, 4, ax=ax1)
    #plt.xlabel('Altimetry, backscatter (dB)')
    plt.xlabel('Altimetry dh/dt (m/yr)')
    plt.ylabel('Firn model dh/dt (m/yr)')
    plt.xlim(-10, 10)
    plt.ylim(-2, 2)
    #plt.savefig('scatter_bs.png', bbox_inches='tight')
    ###
    ax2 = plt.subplot(122)
    corrcoef = np.corrcoef(df_alt.mean(axis=1), df_firn.mean(axis=1))[0,1]
    plt.scatter(df_alt.mean(axis=1), df_firn.mean(axis=1), s=70, marker='o', color='b', alpha=.8)
    plt.title('Average time series')
    ap.intitle('correlation = %.2f' % corrcoef, 4, ax=ax2)
    #plt.xlabel('Altimetry, backscatter (dB)')
    plt.xlabel('Altimetry dh/dt (m/yr)')
    #plt.ylabel('Firn model, elevation (m)')
    #plt.savefig('scatter_bs_avg.png', bbox_inches='tight')
    plt.savefig('altxfirn.png', bbox_inches='tight')
    plt.show()

    # correlation of trends (m/yr)
    '''
    years = ap.date2year(df_alt.index)
    m_alt = np.polyfit(years, df_alt.values, deg=1)[0,:]
    m_firn = np.polyfit(years, df_firn.values, deg=1)[0,:]
    corrcoef = np.corrcoef(m_alt, m_firn)[0,1]
    plt.scatter(m_alt, m_firn, marker='o', color='0.4')
    plt.title('Individual trends for all ice shelves')
    ap.intitle('correlation = %.2f' % corrcoef, 4)
    #plt.xlabel('Altimetry, dh/dt (m/yr)')
    plt.xlabel('Altimetry, dg/dt (dB/yr)')
    plt.ylabel('Firn model, dh/dt (m/yr)')
    #plt.savefig('scatter_dgdt.png', bbox_inches='tight')
    plt.show()
    '''
