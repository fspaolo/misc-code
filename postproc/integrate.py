"""
Integrate grid-cell rates, and calculate full-ice-shelf rates and volumes


Hi Fernando,

I ran two density models on BEDMAP-2 elevations;

Model-A: up to 40 m of firn at 500 kg/m^3 (thick and light, suited to RIS/FRIS);

Model-B: up to 15 m of firn at 700 kg/m^3 (thin and heavy, Ted's suggestion for Wilkins).

Results are (value pairs are mean density and mean thickness)

Model-A:

Firn density    = 500
Max. firn depth = 40
Wilkins
  851.5371  247.6874

Brunt
  856.4734  302.6383

Larsen D
  854.6196  257.6660

Model-B:

Firn density    = 700
Max. firn depth = 15
Wilkins
  899.9239  315.4988

Brunt
  907.0995  408.9416

Larsen D
  899.1307  324.2276

Wilkins values are different from earlier today because there I used my own gridded elevation; here I use BEDMAP-2.

I'd regard these two models as bracketing the likely values for these three ice shelves. Model-A produces ice thicknesses for RIS and FRIS that agree well with B-2.  Model-B is probably good for Wilkins.  Climatically, Brunt and Larsen-D probably fall somewhere between Wilkins and FRIS.

--Laurie

"""

import sys
import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import altimpy as ap
from mpl_toolkits.basemap import interp
from astropy.convolution import convolve, Gaussian2DKernel

FILE_IN = '/Users/fpaolo/data/shelves/'+'h_trendfit.h5'
FILE_OUT = '/Users/fpaolo/data/shelves/'+'h_integrate.csv'
FILE_AREA = '/Users/fpaolo/data/shelves/'+'area_grid_cells.h5'
FILE_DENS = '/Users/fpaolo/data/shelves/'+'density_grid_cells.h5'
FILE_MASK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'

keys = [
    'brunt',
    'riiser',
    'fimbul',
    'lazarev',
    'baudouin',
    'harald',
    'amery',
    'west',
    'shackleton',
    'totten',
    'moscow',
    'holmes',
    'dibble',
    'mertz',
    'cook',
    'rennick',
    'mariner',
    'drygalski',
    'rosse',
    'rossw',
    'sulzberger',
    'nickerson',
    'getz',
    'dotson',
    'crosson',
    'thwaites',
    'pig',
    'cosgrove',
    'abbot',
    'venable',
    'stange',
    'bach',
    'wilkins',
    'georgevi',
    'larsenb',
    'larsenc',
    'larsend',
    'ronne',
    'filchner',
    #
    'qml',
    'wvl',
    'ris',
    'as',
    'bs',
    'lis',
    'fris',
    #
    'eais',
    'wais',
    'ais'
    ]

names = [
    'Brunt',
    'Riiser',
    'Fimbul',
    'Lazarev',
    'Baudouin',
    'Prince Harald',
    'Amery',
    'West',
    'Shackleton',
    'Totten',
    'Moscow',
    'Holmes',
    'Dibble',
    'Mertz',
    'Cook',
    'Rennick',
    'Mariner',
    'Drygalski',
    'Ross EAIS',
    'Ross WAIS',
    'Sulzberger',
    'Nickerson',
    'Getz',
    'Dotson',
    'Crosson',
    'Thwaites',
    'Pine Island',
    'Cosgrove',
    'Abbot',
    'Venable',
    'Stange',
    'Bach',
    'Wilkins',
    'George VI',
    'Larsen B',
    'Larsen C',
    'Larsen D',
    'Ronne',
    'Filchner',
    #
    'Queen Maud',
    'Wilkes-Victoria',
    'Ross',
    'Amundsen',
    'Bellingshausen',
    'Larsen',
    'Filchner-Ronne',
    #
    'East Antarctica',
    'West Antarctica',
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
fin = tb.open_file(FILE_IN)  # data (m)
fa = tb.open_file(FILE_AREA) # area (km^2)
fd = tb.open_file(FILE_DENS) # density (kg/m^3)
fm = tb.open_file(FILE_MASK) # mask
time = fin.root.time[:]
lon = fin.root.lon[:]
lat = fin.root.lat[:]

rate_poly = fin.root.poly_lasso_rate_ib_sl[:]
rate_poly_err = np.abs(fin.root.poly_lasso_rate_ib_sl_err[:])
rate_line = fin.root.line_lstsq_rate_ib_sl[:]
rate_line_err = np.abs(fin.root.line_lstsq_rate_ib_sl_err[:])
rate_dpoly = fin.root.dpoly_lasso_rate[:]
rate_dpoly_err = np.abs(fin.root.dpoly_lasso_rate_error[:])  # FIXME Change name: error -> err

area = fa.root.area[:] * 1e6 # km^2 -> m^2
freeboard = fd.root.freeboard[:]
thickness = fd.root.thickness[:]
thickness_err = fd.root.thickness_err[:]
density = fd.root.density[:]
density_err = fd.root.density_err[:]
mask = fm.root.mask[::2,::2]
x_mask = fm.root.x[::2]
y_mask = fm.root.y[::2]
ny, nx = rate_poly.shape

fin.close()
fa.close()
fd.close()
fm.close()

if 0: # for testing only
    FILE_GL = '/Users/fpaolo/data/coastline/antarctica_gl_ll.h5'
    FILE_COAST = '/Users/fpaolo/data/coastline/moa_coastfile_ll.h5'
    FILE_ISL = '/Users/fpaolo/data/coastline/moa_islands_ll.h5'

    data = rate_poly
    data = area
    ind = ap.where_isnan('cook', lon, lat)
    data[ind] = np.nan
    
    # plot boundaries
    f = tb.open_file(FILE_COAST)
    d = f.root.data[::10]
    y = d[:,0]
    x = ap.lon_180_360(d[:,1])
    f2 = tb.open_file(FILE_GL)
    d2 = f2.root.data[::10]
    y2 = d2[:,0]
    x2 = ap.lon_180_360(d2[:,1])
    f3 = tb.open_file(FILE_ISL)
    d3 = f3.root.data[::10]
    y3 = d3[:,0]
    x3 = ap.lon_180_360(d3[:,1])
    f.close()
    f2.close()
    f3.close()
    
    plt.imshow(data, origin='lower', interpolation='nearest', 
               extent=(lon.min(), lon.max(), lat.min(), lat.max()), 
               aspect='auto')
    plt.plot(x, y, 'ok', markersize=1)
    plt.plot(x2, y2, 'ok', markersize=1)
    plt.plot(x3, y3, 'ok', markersize=1)
    plt.xlim(lon.min(), lon.max())
    plt.ylim(lat.min(), lat.max())
    plt.grid(True)
    plt.show()
    exit()

#------------------------------------------------------

if 1: # smooth without interpolating!
    ind = np.where(np.isnan(rate_poly))
    gauss_kernel = Gaussian2DKernel(1)
    rate_poly = convolve(rate_poly, gauss_kernel, boundary='wrap', normalize_kernel=True)
    rate_poly_err = convolve(rate_poly_err, gauss_kernel, boundary='wrap', normalize_kernel=True)
    rate_line = convolve(rate_line, gauss_kernel, boundary='wrap', normalize_kernel=True)
    rate_line_err = convolve(rate_line_err, gauss_kernel, boundary='wrap', normalize_kernel=True)
    rate_dpoly = convolve(rate_dpoly, gauss_kernel, boundary='wrap', normalize_kernel=True)
    rate_dpoly_err = convolve(rate_dpoly_err, gauss_kernel, boundary='wrap', normalize_kernel=True)
    rate_poly[ind] = np.nan
    rate_poly_err[ind] = np.nan
    rate_line[ind] = np.nan
    rate_line_err[ind] = np.nan
    rate_dpoly[ind] = np.nan
    rate_dpoly_err[ind] = np.nan


    '''
    ind1 = np.where(np.isnan(density))
    ind2 = np.where((density < 700) | (freeboard/thickness > 0.2))
    freeboard[ind2] = np.nan
    thickness[ind2] = np.nan
    density[ind2] = np.nan
    gauss_kernel = Gaussian2DKernel(4)
    freeboard = convolve(freeboard, gauss_kernel, boundary='wrap', normalize_kernel=True)
    thickness = convolve(thickness, gauss_kernel, boundary='wrap', normalize_kernel=True)
    density = convolve(density, gauss_kernel, boundary='wrap', normalize_kernel=True)
    freeboard[ind1] = np.nan
    thickness[ind1] = np.nan
    density[ind1] = np.nan
    '''

if 1: # apply mask to data (some are redundant)
    xx, yy = np.meshgrid(lon, lat)
    xx_polar, yy_polar = ap.ll2xy(xx, yy, units='m')
    mask_rate = interp(mask[::-1,:], x_mask, y_mask[::-1], xx_polar, yy_polar, order=0) # nn
    mask_rate = mask_rate.astype('f8')
    # remove everything but ice shelves
    mask_rate[mask_rate!=4] = np.nan    
    # remove grid cells outside satellite limit
    i, = np.where(lat < -81.6)          
    mask_rate[i,:] = np.nan
    # mask out data
    i, j = np.where(np.isnan(mask_rate))
    rate_poly[i,j] = np.nan
    rate_poly_err[i,j] = np.nan
    rate_line[i,j] = np.nan
    rate_line_err[i,j] = np.nan
    rate_dpoly[i,j] = np.nan
    rate_dpoly_err[i,j] = np.nan
    area[i,j] = np.nan
    freeboard[i,j] = np.nan
    thickness[i,j] = np.nan
    thickness_err[i,j] = np.nan
    density[i,j] = np.nan
    density_err[i,j] = np.nan

if 0:  # save the smoothed data
    pass # => save data here if only smoothed fields needed <=

# save header to csv file
fout = open(FILE_OUT, 'w')
fout.write('ice_shelf,total_area(km2),survey_area(km2),'
           'density(kg/m3),density_err(kg/m3),'
           'thickness(m),thickness_err(m),H/Z,'
           'dhdt_poly(cm/yr),dhdt_poly_err(cm/yr),'
           'dhdt_line(cm/yr),dhdt_line_err(cm/yr),'
           'dmdt_poly(Gt/yr),dmdt_poly_err(Gt/yr),'
           'dmdt_line(Gt/yr),dmdt_line_err(Gt/yr),'
           'accel_poly(cm/yr2),accel_poly_err(cm/yr2),'
           '\n')

# calculate for each ice shelf
for key, name in zip(keys, names):

    rate_poly_ = rate_poly.copy()
    rate_poly_err_ = rate_poly_err.copy()
    rate_line_ = rate_line.copy()
    rate_line_err_ = rate_line_err.copy()
    rate_dpoly_ = rate_dpoly.copy()
    rate_dpoly_err_ = rate_dpoly_err.copy()
    area_ = area.copy()
    freeboard_ = freeboard.copy()
    thickness_ = thickness.copy()
    thickness_err_ = thickness_err.copy()
    density_ = density.copy()
    density_err_ = density_err.copy()

    ind = ap.where_isnan(key, lon, lat)

    rate_poly_[ind] = np.nan
    rate_poly_err_[ind] = np.nan
    rate_line_[ind] = np.nan
    rate_line_err_[ind] = np.nan
    rate_dpoly_[ind] = np.nan
    rate_dpoly_err_[ind] = np.nan
    area_[ind] = np.nan
    freeboard_[ind] = np.nan
    thickness_[ind] = np.nan
    thickness_err_[ind] = np.nan
    density_[ind] = np.nan
    density_err_[ind] = np.nan

    # area-weighted sum -> average
    #-----------------------------

    sampled_area = area_.copy()                      # full area
    sampled_area[np.isnan(rate_poly_)] = np.nan      # sampled area
    samp_area = np.nansum(sampled_area)              # total sampled area (m^2)
    total_area = np.nansum(area_)                    # total area (m^2)

    # sampled-area weights
    weight = sampled_area
    weight /= np.nansum(weight)                      # normalize sum(w) = 1

    height_rate = np.nansum(weight * rate_poly_)     # m/yr
    height_rate2 = np.nansum(weight * rate_line_)
    height_accel = np.nansum(weight * rate_dpoly_)   # cm/yr2

    height_rate_err = np.sqrt(np.nansum(weight**2 * rate_poly_err_**2))
    height_rate_err2 = np.sqrt(np.nansum(weight**2 * rate_line_err_**2))
    height_accel_err = np.sqrt(np.nansum(weight**2 * rate_dpoly_err_**2))

    # full-area weights
    weight = area_
    weight /= np.nansum(weight)                       # normalize sum(w) = 1

    # FIXME What about the uncertainties?
    if name == 'Wilkins':
        rho = np.array(899.9239) 
        thick = np.array(315.4988)
        ratio = 1 / (1 - rho / 1028.)  # use water density
    elif name == 'Brunt':
        rho = np.array(856.4734)
        thick = np.array(302.6383)
        ratio = 1 / (1 - rho / 1028.)
    elif name == 'Larsen D':
        rho = np.array(854.6196)
        thick = np.array(257.6660)
        ratio = 1 / (1 - rho / 1028.)
    else:
        rho = np.nansum(weight * density_)                # kg/m^3 
        thick = np.nansum(weight * thickness_)            # m
        ratio = np.nansum(weight * (thickness_/freeboard_))

    rho_err = np.sqrt(np.nansum(weight**2 * density_err_**2))
    thick_err = np.sqrt(np.nansum(weight**2 * thickness_err_**2))

    # Mass rate -> m' = area * rho * Z/H * h' (all average values)
    #-------------------------------------------------------------

    # FIXME Mass errors too small???

    # full area x full density x full ratio x sampled rates
    mass_rate =  total_area * rho * ratio * height_rate  # kg/yr
    mass_rate2 = total_area * rho * ratio * height_rate2

    mass_rate_err = mass_rate * np.sqrt((rho_err/rho + thick_err/thick)**2 + \
                                        (height_rate_err/height_rate)**2)
    mass_rate_err2 = mass_rate2 * np.sqrt((rho_err/rho + thick_err/thick)**2 + \
                                          (height_rate_err2/height_rate2)**2)

    # convert units
    #--------------------------------------------------

    height_rate *= 1e2              # m/yr -> cm/yr
    height_rate2 *= 1e2 
    height_rate_err *= 1e2 
    height_rate_err2 *= 1e2
    height_accel *= 1e2
    height_accel_err *= 1e2
    mass_rate *= 0.001 * 1e-9       # kg/yr -> Gt/yr
    mass_rate2 *= 0.001 * 1e-9
    mass_rate_err *= 0.001 * 1e-9
    mass_rate_err2 *= 0.001 * 1e-9
    samp_area *= 1e-6               # m^2 -> km^2
    total_area *= 1e-6 

    if 1:  # scale and truncate results
        height_rate_err *= 3
        height_rate_err2 *= 3
        height_accel_err *= 3
        mass_rate_err *= 3
        mass_rate_err2 *= 3

        # precision -> 0.05 cm, 0.05 Gt/yr, 0.05 cm/yr2
        height_rate = height_rate.round(2)
        height_rate_err = height_rate_err.round(2)
        height_rate2 = height_rate2.round(2)
        height_rate_err2 = height_rate_err2.round(2)
        height_accel = height_accel.round(2)
        height_accel_err = height_accel_err.round(2)
        mass_rate = mass_rate.round(2)
        mass_rate_err = mass_rate_err.round(2)
        mass_rate2 = mass_rate2.round(2)
        mass_rate_err2 = mass_rate_err2.round(2)
        rho = rho.round(2)
        thick = thick.round(2)

        # force no mass change if height is less than the precision
        if np.abs(height_rate) < 0.05:
            mass_rate = 0.0
        if np.abs(height_rate2) < 0.05:
            mass_rate2 = 0.0

        # force min error to be half of the precision (non-zero)
        if height_rate_err < 0.05:
            height_rate_err = 0.05
        if height_rate_err2 < 0.05:
            height_rate_err2 = 0.05
        if mass_rate_err < 0.05:
            mass_rate_err = 0.05
        if mass_rate_err2 < 0.05:
            mass_rate_err2 = 0.05
        if height_accel_err < 0.05:
            height_accel_err = 0.05

    if 1:
        print name
        print 'total area:', total_area, 'km2'
        print 'survey area:', samp_area, 'km2'
        print "h': %.1f (%.1f) +/- %.1f (%.1f) cm/yr  " % \
                (height_rate, height_rate2, height_rate_err, height_rate_err2)
        print "m':  %.1f (%.1f) +/- %.1f (%.1f) Gt/yr" % \
                (mass_rate, mass_rate2, mass_rate_err, mass_rate_err2)
        print "h'': %.1f +/- %.1f cm/yr2" % (height_accel, height_accel_err)
        print '-'*32

    # save data 
    #--------------------------------------------------

    if 1:
        fout.write('%s,%g,%g,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,' \
                   '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' %
                   (name, total_area, samp_area, rho, rho_err, thick, 
                    thick_err, 1/ratio, height_rate, height_rate_err,
                    height_rate2, height_rate_err2, mass_rate, mass_rate_err,
                    mass_rate2, mass_rate_err2, height_accel, height_accel_err,
                    )
                   )

fout.close()
print 'done'
print 'out -> ', FILE_OUT
