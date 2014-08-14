"""
Fit polynomials to the time series using least-squares and cross-validation.

"""

import sys
import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import altimpy as ap


PLOT = True
#FILE_IN = 'all_19920716_20111015_shelf_tide_grids_mts.h5.ice_oce'
FILE_IN = 'h_postproc.h5'
FILE_OUT = 'h_trendfit.h5'
DIR = '/Users/fpaolo/data/shelves/' 


#####################################################################
### CHECK LARSENC TIME SERIES I=4, J=4 @ 2003 !!! XCALIB PROBLEM? ###
#####################################################################

def rate(y, x=None):
    """Average rate of change."""
    return (y[-1] - y[0]) / (x[-1] - x[0])


def rate_err(x, yerr, independent=True):
    """Uncertainty for rate of change."""
    if independent:
        err = (np.sqrt(2) * yerr) / (x[-1] - x[0])  # independent errors
    else:
        err = (2 * yerr) / (x[-1] - x[0])           # dependent errors
    return err


def lasso_cv(y, x=None, max_deg=3):
    """LASSO fit."""
    if np.isnan(y).all():
        y_pred = y
    else:
        y_pred = ap.lasso_cv(x, y, cv=10, max_deg=max_deg, max_iter=1e3)
    return y_pred


def lstsq_cv(y, x=None, max_deg=3):
    """LSTSQ fit."""
    if np.isnan(y).all():
        y_pred = y
    else:
        y_pred = ap.lstsq_cv(x, y, cv=10, max_deg=max_deg)
    return y_pred


def line_fit(y, x=None):
    y_pred = y
    ind, = np.where(~np.isnan(y))
    if len(ind) > 2:
        y_, x_ = y[ind], x[ind]
        a = np.polyfit(x_, y_, 1)
        y_pred = np.polyval(a, x)
    return y_pred


def gradient(y, dt=.25):
    """Central difference."""
    return np.gradient(y, dt)


def gradient_err(x, yerr, independent=True):
    """Error for central difference."""
    h = x[2] - x[1]
    if independent: 
        err = (np.sqrt(2) * yerr) / (2 * h)
    else:
        err = yerr / h  # 2*e / 2*h
    return err


def as_frame(data, z, y, x):
    """3d Array -> Data Frame."""
    try:
        return pd.Panel(data, items=z, major_axis=y, minor_axis=x
                        ).to_frame(filter_observations=False).T
    except:
        print 'already a DataFrame'
        return data


def as_array(data):
    """Data Frame -> 3d Array."""
    try:
        return data.T.to_panel().values
    except:
        print 'already an Array'
        return data


def write_slabs(fid, var_name, data):
    """Save 3d array into several slabs, for XDMF."""
    g = fid.create_group('/', var_name + '_slabs')
    for i, d in enumerate(data):
        fid.create_array(g, var_name +'_%02d' % i, d)

#------------------------------------------------------

print 'loading data...'
fin = tb.open_file(DIR + FILE_IN)
try:
    time = ap.num2year(fin.root.time[:])
except:
    time = fin.root.time[:]
lon = fin.root.lon[:]
lat = fin.root.lat[:]
data = fin.root.dh_mean_mixed_const_xcal[:]
error = fin.root.dh_error_xcal[:]
nz, ny, nx = data.shape


if 0: # subset
    print 'subsetting...'
    region = ap.pig
    data, _, _ = ap.get_subset(region, data, lon, lat)
    error, lon, lat = ap.get_subset(region, error, lon, lat)
    nt, ny, nx = data.shape                # i,j,k = t,y,x
    print 'done'

#------------------------------------------------------

if 0: # plot alphas and MSEs (the prediction error curve)
    N = 3
    x = time
    y = data[:,5,2] # 6,2 -> PIG
    y = ap.referenced(y, to='mean')
    y_pred, lasso = ap.lasso_cv(x, y, cv=10, max_deg=N, max_iter=1e3, return_model=True)
    mse = lasso.mse_path_.mean(axis=1)
    std = lasso.mse_path_.std(axis=1, ddof=1) / np.sqrt(10)
    #plt.plot(np.log(lasso.alphas_), mse)
    plt.errorbar(np.log(lasso.alphas_), mse, yerr=std)
    plt.vlines(np.log(lasso.alpha_), ymin=mse.min(), ymax=mse.max(), color='r')
    plt.xlabel('log(alpha)')
    plt.ylabel('10-fold-average MSE')
    plt.show()
    exit()

#------------------------------------------------------

data = as_frame(data, time, lat, lon)
error = as_frame(error, time, lat, lon)
data = data.apply(ap.referenced, to='mean', raw=True)

# polynomial fit
poly = data.apply(lasso_cv, x=time, max_deg=3, raw=True)
poly2 = data.apply(lstsq_cv, x=time, max_deg=3, raw=True)

# polynomial error
poly_err = ap.rmse(poly, data)   # rmse < ser !    TODO: decide on RMSE vs GSE
#poly_err = ap.gse(poly, data)     # global std err
poly_err2 = ap.gse(poly, data)

# polynomial average rate
poly_rate = poly.apply(rate, x=time, raw=True)
poly2_rate = poly2.apply(rate, x=time, raw=True)

# polynomial average rate error
poly_rate_err = rate_err(time, poly_err, independent=False)

# line fit
line = data.apply(line_fit, x=time, raw=True)

# line fit error
line_err = ap.rmse(line, data)                   # TODO: decide on RMSE vs GSE
#line_err = ap.gse(line, data)

# line average rate
line_rate = data.apply(rate, x=time, raw=True)

# line average rate error
line_rate_err = rate_err(time, line_err, independent=False)

# derivative
dpoly = poly.apply(gradient,  dt=time[2]-time[1], raw=True)

# derivative error -> FIXME something wrong with the propagation!
#dpoly_err = gradient_err(time, poly_err, independent=False)
dpoly_err = gradient_err(time, ap.gse(poly, data), independent=False)

# derivative average rate
dpoly_rate = dpoly.apply(rate, x=time, raw=True)

# derivative average rate error -> FIXME something wrong with the propagation!
dpoly_rate_err = rate_err(time, dpoly_err, independent=False)

if 0:
    poly = poly.apply(ap.referenced, to='mean', raw=True)
    dpoly = dpoly.apply(ap.referenced, to='mean', raw=True)

#------------------------------------------------------

if 0: # plot error bars
    poly_err *= 3
    for i, c in enumerate(poly.columns):
        if poly[c].notnull().any():
            print 'GSE:', poly_err[i].round(2)
            #print 'MSE:', poly_err2[i].round(2)
            data[c].plot()
            plt.errorbar(poly.index, poly[c], yerr=poly_err[i])
            #plt.errorbar(poly.index, poly[c], yerr=poly_err2[i])
            plt.show()
    exit()

#------------------------------------------------------

# 1d(2d) -> 2d(3d)
poly = as_array(poly)
poly_err = poly_err.reshape(ny,nx)
poly_rate = poly_rate.reshape(ny,nx) 
poly_rate_err = poly_rate_err.reshape(ny,nx)

poly2 = as_array(poly2)
poly2_rate = poly2_rate.reshape(ny,nx) 

line = as_array(line)
line_err = line_err.reshape(ny,nx)
line_rate = line_rate.reshape(ny,nx)
line_rate_err = line_rate_err.reshape(ny,nx)

dpoly = as_array(dpoly)
dpoly_err = dpoly_err.reshape(ny,nx)
dpoly_rate = dpoly_rate.reshape(ny,nx) 
dpoly_rate_err = dpoly_rate_err.reshape(ny,nx)

#------------------------------------------------------

if 0: # plot maps
    plt.figure()
    plt.imshow(poly_rate, origin='lower', interpolation='nearest', vmin=-.1, vmax=.1)
    plt.figure()
    plt.imshow(line_rate, origin='lower', interpolation='nearest', vmin=-.1, vmax=.1)
    exit()

if 0: # print errors
    f = poly_rate[~np.isnan(poly_rate)].round(2)
    e = poly_rate_err[~np.isnan(poly_rate_err)].round(2)
    print '[poly]'
    print 'rate (m/yr):  ', f 
    print 'error (m/yr): ', e
    print 'frac error:   ', (np.abs(e/f) * 100).round(1)
    f = line_rate[~np.isnan(line_rate)].round(2)
    e = line_rate_err[~np.isnan(line_rate_err)].round(2)
    print '[line]'
    print 'rate (m/yr):  ', f 
    print 'error (m/yr): ', e
    print 'frac error:   ', (np.abs(e/f) * 100).round(1)
    f = dpoly_rate[~np.isnan(dpoly_rate)].round(2)
    e = dpoly_rate_err[~np.isnan(dpoly_rate_err)].round(2)
    print '[dpoly]'
    print 'acce (m/yr2): ', f
    print 'error (m/yr2):', e
    print 'frac error:   ', (np.abs(e/f) * 100).round(1)
    exit()

#---------------------------------------------------------------------
# save data 
#---------------------------------------------------------------------

# spherical -> cartesian
xx, yy = np.meshgrid(lon, lat)
xed, yed = ap.cell2node(lon, lat)
xed2d, yed2d = np.meshgrid(xed, yed)
xyz = np.column_stack(ap.sph2xyz(xed2d.ravel(), yed2d.ravel()))

if 1:
    print('saving data...')
    fout = tb.open_file(DIR + FILE_OUT, 'w')
    try:
        fout.create_array('/', 'time', time)
    except:
        pass
    try:
        fout.create_array('/', 'lon', lon)
        fout.create_array('/', 'lat', lat)
    except:
        pass
    try:
        fout.create_array('/', 'xyz_nodes', xyz)
    except:
        pass
    try:
        fout.create_array('/', 'xx', xx)
        fout.create_array('/', 'yy', yy)
    except:
        pass
    # EDIT HERE
    fout.create_array('/', 'poly_lasso', poly)  # 3d
    fout.create_array('/', 'poly_lasso_error', poly_err)  # 3d
    fout.create_array('/', 'poly_lasso_rate', poly_rate)  # 2d
    fout.create_array('/', 'poly_lasso_rate_error', poly_rate_err)  # 2d

    fout.create_array('/', 'poly_lstsq', poly2)
    fout.create_array('/', 'poly_lstsq_rate', poly2_rate)

    #fout.create_array('/', 'line_lstsq', line)
    fout.create_array('/', 'line_lstsq_rate', line_rate)
    fout.create_array('/', 'line_lstsq_rate_error', line_rate_err)

    fout.create_array('/', 'dpoly_lasso', dpoly)
    fout.create_array('/', 'dpoly_lasso_error', dpoly_err)
    fout.create_array('/', 'dpoly_lasso_rate', dpoly_rate)
    fout.create_array('/', 'dpoly_lasso_rate_error', dpoly_rate_err)

    fout.flush()
    fout.close()
    print 'done'
    print 'out -> ' + DIR + FILE_OUT

fin.close()
