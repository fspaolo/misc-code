"""
Integrate grid-cell rates, and calculate full-ice-shelf rates and volumes

"""

import sys
import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import altimpy as ap
from mpl_toolkits.basemap import interp
from astropy.convolution import convolve, Gaussian2DKernel


PLOT = True
FILE_IN = '/Users/fpaolo/data/shelves/'+'h_trendfit.h5'
FILE_OUT = '/Users/fpaolo/data/shelves/'+'h_integrate.csv'
FILE_AREA = '/Users/fpaolo/data/shelves/'+'area_grid_cells.h5'
FILE_MASK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'

regions = [
    ap.fimbulw,
    ap.fimbule,
    ap.lazarev,
    ap.amery,
    ap.west,
    ap.shackleton,
    ap.totten,
    ap.moscow,
    ap.rosse,
    ap.rossw,
    ap.sulzberger,
    ap.getz,
    ap.dotson,
    ap.crosson,
    ap.thwaites,
    ap.pig,
    ap.abbot,
    ap.stange,
    ap.bach,
    ap.wilkins,
    ap.georges,
    ap.georgen,
    ap.larsenb,
    ap.larsenc,
    ap.larsend,
    ap.ronne,
    ap.filchner,
    ap.brunt,
    ap.riiser,
    #
    ap.queenmaud,
    ap.fris,
    ap.larsen,
    ap.belling,
    ap.amundsen,
    ap.ris,
    ap.tottmosc,
    ap.westshac,
    ap.wais,
    None, #ap.eais,
    ap.ais,
    ]

names = [
    'Fimbul W',
    'Fimbul E',
    'Lazarev',
    'Amery',
    'West',
    'Shackleton',
    'Totten',
    'Moscow',
    'Ross E',
    'Ross W',
    'Sulzberger',
    'Getz',
    'Dotson',
    'Crosson',
    'Thwaites',
    'Pine Island',
    'Abbot',
    'Stange',
    'Bach',
    'Wilkins',
    'George S',
    'George N',
    'Larsen B',
    'Larsen C',
    'Larsen D',
    'Ronne',
    'Filchner',
    'Brunt',
    'Riiser',
    #
    'Queen Maud',
    'Filchner Ronne',
    'Eastern AP',
    'Bellingshausen',
    'Amundsen',
    'Ross',
    'Totten Moscow',
    'West Shackleton',
    'West Antarctica',
    'East Antarctica',
    'All Antarctica',
    ]


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

#------------------------------------------------------

print 'loading data...'
fin = tb.open_file(FILE_IN)  # data
fa = tb.open_file(FILE_AREA) # area
fm = tb.open_file(FILE_MASK) # mask
time = fin.root.time[:]
lon = fin.root.lon[:]
lat = fin.root.lat[:]
rate_poly = fin.root.poly_lasso_rate[:] * 1e2  # m -> cm
error_poly = np.abs(fin.root.poly_lasso_rate_error[:]) * 1e2 # TODO, abs needed?
rate_line = fin.root.line_lstsq_rate[:] * 1e2
error_line = np.abs(fin.root.line_lstsq_rate_error[:]) * 1e2
rate_dpoly = fin.root.dpoly_lasso_rate[:] * 1e2
error_dpoly = np.abs(fin.root.dpoly_lasso_rate_error[:]) * 1e2
area = fa.root.area[:]
mask = fm.root.mask[::2,::2]
mask_x = fm.root.x[::2]
mask_y = fm.root.y[::2]
ny, nx = rate_poly.shape
fin.close()
fa.close()
fm.close()


if 0: # subset
    print 'subsetting...'
    region = ap.larsen
    rate_poly, _, _ = ap.get_subset(region, rate_poly, lon, lat)
    error_poly, _, _ = ap.get_subset(region, error_poly, lon, lat)
    rate_line, _, _ = ap.get_subset(region, rate_line, lon, lat)
    error_line, _, _ = ap.get_subset(region, error_line, lon, lat)
    rate_dpoly, _, _ = ap.get_subset(region, rate_dpoly, lon, lat)
    error_dpoly, _, _ = ap.get_subset(region, error_dpoly, lon, lat)
    area, lon, lat = ap.get_subset(region, area, lon, lat)
    ny, nx = rate_poly.shape
    print 'done'

#------------------------------------------------------

if 0:
    plt.figure()
    plt.imshow(rate_poly, origin='lower', interpolation='nearest')
    plt.figure()
    plt.imshow(error_poly, origin='lower', interpolation='nearest')

ij_samp = np.where(np.isnan(rate_poly))  # original data coverage

if 1: # smooth and interpolate before integrating
    gauss_kernel = Gaussian2DKernel(1)  # std of 1 pixel
    rate_poly = convolve(rate_poly, gauss_kernel, boundary='wrap', normalize_kernel=True)
    error_poly = convolve(error_poly, gauss_kernel, boundary='wrap', normalize_kernel=True)
    rate_line = convolve(rate_line, gauss_kernel, boundary='wrap', normalize_kernel=True)
    error_line = convolve(error_line, gauss_kernel, boundary='wrap', normalize_kernel=True)
    rate_dpoly = convolve(rate_dpoly, gauss_kernel, boundary='wrap', normalize_kernel=True)
    error_dpoly = convolve(error_dpoly, gauss_kernel, boundary='wrap', normalize_kernel=True)

if 1: # apply mask to data
    xx, yy = np.meshgrid(lon, lat)
    xx_polar, yy_polar = ap.ll2xy(xx, yy, units='m')
    mask_rate = interp(mask[::-1,:], mask_x, mask_y[::-1], xx_polar, yy_polar, order=0) # nn
    mask_rate = mask_rate.astype('f8')
    # remove everything but ice shelves
    mask_rate[mask_rate!=4] = np.nan    
    # remove grid cells outside satellite limit
    i, = np.where(lat < -81.6)          
    mask_rate[i,:] = np.nan
    # mask out data
    i, j = np.where(np.isnan(mask_rate))
    '''
    plt.figure()
    plt.imshow(rate_poly, origin='lower', interpolation='nearest')
    '''
    rate_poly[i,j] = np.nan
    error_poly[i,j] = np.nan
    rate_line[i,j] = np.nan
    error_line[i,j] = np.nan
    rate_dpoly[i,j] = np.nan
    error_dpoly[i,j] = np.nan
    area[i,j] = np.nan
    '''
    plt.figure()
    plt.imshow(rate_poly, origin='lower', interpolation='nearest')
    plt.show()
    exit()
    '''

# two datasets: 'sampled' and 'full' coverage
rate_poly_samp = rate_poly.copy() 
error_poly_samp = error_poly.copy()
rate_line_samp = rate_line.copy() 
error_line_samp = error_line.copy()
rate_dpoly_samp = rate_dpoly.copy() 
error_dpoly_samp = error_dpoly.copy()
area_samp = area.copy()

rate_poly_samp[ij_samp] = np.nan 
error_poly_samp[ij_samp] = np.nan 
rate_line_samp[ij_samp] = np.nan 
error_line_samp[ij_samp] = np.nan 
rate_dpoly_samp[ij_samp] = np.nan
error_dpoly_samp[ij_samp] = np.nan
area_samp[ij_samp] = np.nan

# save header to csv file
fout = open(FILE_OUT, 'w')
fout.write('ice_shelf,total_area(km2),survey_area(km2),dhdt_poly(cm/yr),'
           'dhdt_poly_err(cm/yr),dhdt_line(cm/yr), dhdt_line_err(cm/yr),'
           'dvdt_poly(km3/yr),dvdt_poly_err(km3/yr),dvdt_line(km3/yr),'
           'dvdt_line_err(km3/yr),accel_poly(cm/yr2),accel_poly_err(cm/yr2)\n')

# calculate for each ice shelf
for name, region in zip(names, regions):

    if name == 'East Antarctica':
        rate_poly_full_1, _, _ = ap.get_subset(ap.eais1, rate_poly, lon, lat)
        error_poly_full_1, _, _ = ap.get_subset(ap.eais1, error_poly, lon, lat)
        rate_line_full_1, _, _ = ap.get_subset(ap.eais1, rate_line, lon, lat)
        error_line_full_1, _, _ = ap.get_subset(ap.eais1, error_line, lon, lat)
        rate_dpoly_full_1, _, _ = ap.get_subset(ap.eais1, rate_dpoly, lon, lat)
        error_dpoly_full_1, _, _ = ap.get_subset(ap.eais1, error_dpoly, lon, lat)
        area_full_1, lon_1, lat_1 = ap.get_subset(ap.eais1, area, lon, lat)

        rate_poly_samp_1, _, _ = ap.get_subset(ap.eais1, rate_poly_samp, lon, lat)
        error_poly_samp_1, _, _ = ap.get_subset(ap.eais1, error_poly_samp, lon, lat)
        rate_line_samp_1, _, _ = ap.get_subset(ap.eais1, rate_line_samp, lon, lat)
        error_line_samp_1, _, _ = ap.get_subset(ap.eais1, error_line_samp, lon, lat)
        rate_dpoly_samp_1, _, _ = ap.get_subset(ap.eais1, rate_dpoly_samp, lon, lat)
        error_dpoly_samp_1, _, _ = ap.get_subset(ap.eais1, error_dpoly_samp, lon, lat)
        area_samp_1, lon_1, lat_1 = ap.get_subset(ap.eais1, area_samp, lon, lat)

        rate_poly_full_2, _, _ = ap.get_subset(ap.eais2, rate_poly, lon, lat)
        error_poly_full_2, _, _ = ap.get_subset(ap.eais2, error_poly, lon, lat)
        rate_line_full_2, _, _ = ap.get_subset(ap.eais2, rate_line, lon, lat)
        error_line_full_2, _, _ = ap.get_subset(ap.eais2, error_line, lon, lat)
        rate_dpoly_full_2, _, _ = ap.get_subset(ap.eais2, rate_dpoly, lon, lat)
        error_dpoly_full_2, _, _ = ap.get_subset(ap.eais2, error_dpoly, lon, lat)
        area_full_2, lon_2, lat_2 = ap.get_subset(ap.eais2, area, lon, lat)

        rate_poly_samp_2, _, _ = ap.get_subset(ap.eais2, rate_poly_samp, lon, lat)
        error_poly_samp_2, _, _ = ap.get_subset(ap.eais2, error_poly_samp, lon, lat)
        rate_line_samp_2, _, _ = ap.get_subset(ap.eais2, rate_line_samp, lon, lat)
        error_line_samp_2, _, _ = ap.get_subset(ap.eais2, error_line_samp, lon, lat)
        rate_dpoly_samp_2, _, _ = ap.get_subset(ap.eais2, rate_dpoly_samp, lon, lat)
        error_dpoly_samp_2, _, _ = ap.get_subset(ap.eais2, error_dpoly_samp, lon, lat)
        area_samp_2, lon_2, lat_2 = ap.get_subset(ap.eais2, area_samp, lon, lat)

        rate_poly_full_ = np.c_[rate_poly_full_1, rate_poly_full_2]
        error_poly_full_ = np.c_[error_poly_full_1, error_poly_full_2]
        rate_line_full_ = np.c_[rate_line_full_1, rate_line_full_2]
        error_line_full_ = np.c_[error_line_full_1, error_line_full_2]
        rate_dpoly_full_ = np.c_[rate_dpoly_full_1, rate_dpoly_full_2]
        error_dpoly_full_ = np.c_[error_dpoly_full_1, error_dpoly_full_2]
        area_full_ = np.c_[area_full_1, area_full_2]

        rate_poly_samp_ = np.c_[rate_poly_samp_1, rate_poly_samp_2]
        error_poly_samp_ = np.c_[error_poly_samp_1, error_poly_samp_2]
        rate_line_samp_ = np.c_[rate_line_samp_1, rate_line_samp_2]
        error_line_samp_ = np.c_[error_line_samp_1, error_line_samp_2]
        rate_dpoly_samp_ = np.c_[rate_dpoly_samp_1, rate_dpoly_samp_2]
        error_dpoly_samp_ = np.c_[error_dpoly_samp_1, error_dpoly_samp_2]
        area_samp_ = np.c_[area_samp_1, area_samp_2]

    else:
        rate_poly_full_, _, _ = ap.get_subset(region, rate_poly, lon, lat)
        error_poly_full_, _, _ = ap.get_subset(region, error_poly, lon, lat)
        rate_line_full_, _, _ = ap.get_subset(region, rate_line, lon, lat)
        error_line_full_, _, _ = ap.get_subset(region, error_line, lon, lat)
        rate_dpoly_full_, _, _ = ap.get_subset(region, rate_dpoly, lon, lat)
        error_dpoly_full_, _, _ = ap.get_subset(region, error_dpoly, lon, lat)
        area_full_, lon_, lat_ = ap.get_subset(region, area, lon, lat)

        rate_poly_samp_, _, _ = ap.get_subset(region, rate_poly_samp, lon, lat)
        error_poly_samp_, _, _ = ap.get_subset(region, error_poly_samp, lon, lat)
        rate_line_samp_, _, _ = ap.get_subset(region, rate_line_samp, lon, lat)
        error_line_samp_, _, _ = ap.get_subset(region, error_line_samp, lon, lat)
        rate_dpoly_samp_, _, _ = ap.get_subset(region, rate_dpoly_samp, lon, lat)
        error_dpoly_samp_, _, _ = ap.get_subset(region, error_dpoly_samp, lon, lat)
        area_samp_, lon_, lat_ = ap.get_subset(region, area_samp, lon, lat)

    # elevation rate (area-weighted sum) -> use sampled area!
    weight = area_samp_.copy()
    weight /= np.nansum(weight)                       # normalize sum(w) = 1
    elev_rate = np.nansum(weight * rate_poly_samp_)   # cm/yr
    elev_rate2 = np.nansum(weight * rate_line_samp_)  # cm/yr

    # elevation error
    elev_err = np.sqrt(np.nansum(weight**2 * error_poly_samp_**2))
    elev_err2 = np.sqrt(np.nansum(weight**2 * error_line_samp_**2))

    # volume rate (area integral) -> use full area!
    rate_poly_full_ *= 1e-5  # cm -> km
    rate_line_full_ *= 1e-5
    vol_rate = np.nansum(area_full_ * rate_poly_full_)  # km3/yr
    vol_rate2 = np.nansum(area_full_ * rate_line_full_) 

    # volume error
    error_poly_full_ *= 1e-5 # cm -> km
    error_line_full_ *= 1e-5
    vol_err = np.sqrt(np.nansum(area_full_**2 * error_poly_full_**2))
    vol_err2 = np.sqrt(np.nansum(area_full_**2 * error_line_full_**2))

    # acceleration -> use sampled area!
    elev_accel = np.nansum(weight * rate_dpoly_samp_)  # cm/yr2

    # acceleration error
    accel_err = np.sqrt(np.nansum(weight**2 * error_dpoly_samp_**2))

    if 1:  # scale and truncate results
        elev_err *= 3
        elev_err2 *= 3
        vol_err *= 3
        vol_err2 *= 3
        accel_err *= 3

        # precision -> 1 mm, .01 km3/yr, 1 mm/yr2
        elev_rate = elev_rate.round(2)
        elev_err = elev_err.round(2)
        elev_rate2 = elev_rate2.round(2)
        elev_err2 = elev_err2.round(2)
        vol_rate = vol_rate.round(2)
        vol_err = vol_err.round(2)
        vol_rate2 = vol_rate2.round(2)
        vol_err2 = vol_err2.round(2)
        elev_accel = elev_accel.round(2)
        accel_err = accel_err.round(2)
        area_full_ = np.nansum(area_full_)
        area_samp_ = np.nansum(area_samp_)

        # force no volume change if elevation is less than the precision
        if np.abs(elev_rate) < 0.05:
            vol_rate = 0.0
        if np.abs(elev_rate2) < 0.05:
            vol_rate2 = 0.0

        # force error to be non-zero -> half of the precision!
        if elev_err < 0.05:
            elev_err = 0.05
        if elev_err2 < 0.05:
            elev_err2 = 0.05
        if vol_err < 0.05:
            vol_err = 0.05
        if vol_err2 < 0.05:
            vol_err2 = 0.05
        if accel_err < 0.05:
            accel_err = 0.05

    if 1:
        print 'shelf:', name
        print 'total area:', int(np.nansum(area_full_)), 'km2'
        print 'survey area:', int(np.nansum(area_samp_)), 'km2'
        print 'elev rate: %.1f (%.1f) +/- %.1f (%.1f) cm/yr  ' % \
                (elev_rate, elev_rate2, elev_err, elev_err2)
        print 'vol rate:  %.1f (%.1f) +/- %.1f (%.1f) km3/yr' % \
                (vol_rate, vol_rate2, vol_err, vol_err2)
        print 'elev accel: %.1f +/- %.1f cm/yr2' % (elev_accel, accel_err)
        print '-'*32

    # save data 
    #--------------------------------------------------

    if 1:
        fout.write('%s,%g,%g,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,' \
                   '%.2f,%.2f\n' %
                   (name, area_full_, area_samp_, elev_rate, elev_err,
                    elev_rate2, elev_err2, vol_rate, vol_err, vol_rate2,
                    vol_err2, elev_accel, accel_err))

fout.close()
print 'done'
print 'out -> ', FILE_OUT
