
import numpy as np
import tables as tb
import altimpy as ap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import interp
from astropy.convolution import convolve, Gaussian2DKernel

FILE_OUT = '/Users/fpaolo/data/shelves/density_grid_cells.h5'
FILE_DHDT = '/Users/fpaolo/data/shelves/h_trendfit.h5'
FILE_MASK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'
FILE_FREEBOARD = '/Users/fpaolo/data/bedmap/bedmap2_bin/bedmap2_surface.flt'
FILE_THICKNESS = '/Users/fpaolo/data/bedmap/bedmap2_bin/bedmap2_thickness.flt'
FILE_THICKNESS_ERR = '/Users/fpaolo/data/bedmap/bedmap2_bin/bedmap2_thickness_uncertainty_5km.flt'

# Bedmap2 data parameters
cols = 6667
rows = 6667
left = -3333500
right = 3333500
bottom = -3333500
top = 3333500
dx = 1000
dy = 1000
nodata = -9999

# Bedmap2 error parameters
cols_err = 1361
rows_err = 1361
left_err = -3401500
right_err = 3401500
bottom_err = -3402500
top_err = 3402500
dx_err = 5000
dy_err = 5000
nodata = -9999

density_water = 1028.

dsample = 2

# read data <- HDF5
f = tb.open_file(FILE_DHDT)
dhdt = f.root.poly_lasso_rate[:]
lon = f.root.lon[:]
lat = f.root.lat[:]
f.close()

# read Mask <- HDF5
f = tb.open_file(FILE_MASK, 'r')
x_mask = f.root.x[::dsample]
y_mask = f.root.y[::dsample]
mask = f.root.mask[::dsample,::dsample]
mask = mask[::-1]   # flip y-dim
y_mask = y_mask[::-1]
f.close()

# read Bedmap2 <- Bin
freeboard = np.fromfile(FILE_FREEBOARD, dtype='<f4').reshape(rows, cols)[::dsample,::dsample]
thickness = np.fromfile(FILE_THICKNESS, dtype='<f4').reshape(rows, cols)[::dsample,::dsample]
x_bm = np.linspace(left, right, cols)[::dsample]
y_bm = np.linspace(bottom, top, rows)[::dsample]

thickness_err = np.fromfile(FILE_THICKNESS_ERR, dtype='<f4').reshape(rows_err, cols_err)
x_err = np.linspace(left_err, right_err, cols_err)
y_err = np.linspace(bottom_err, top_err, rows_err)

freeboard = freeboard[::-1]  # flip y-dim
thickness = thickness[::-1]
thickness_err = thickness_err[::-1]
ind = np.where((freeboard == nodata) | (thickness == nodata))
freeboard[ind] = np.nan
thickness[ind] = np.nan
thickness_err[thickness_err==nodata] = np.nan

# grid coords in polar stere
xx, yy = np.meshgrid(lon, lat)
xx, yy = ap.ll2xy(xx, yy, units='m')
xx_bm, yy_bm = np.meshgrid(x_bm, y_bm)
xx_mask, yy_mask = np.meshgrid(x_mask, y_mask)

# regrid the error to match resolutions
thickness_err1 = interp(thickness_err, x_err, y_err, xx_bm, yy_bm, order=1)
thickness_err2 = interp(thickness_err, x_err, y_err, xx_bm, yy_bm, order=0)
thickness_err1[np.isnan(thickness_err1)] = thickness_err2[np.isnan(thickness_err1)]
thickness_err = thickness_err1

# mask ice shelves 
mask_bm = interp(mask, x_mask, y_mask, xx_bm, yy_bm, order=0)
freeboard[mask_bm!=4] = np.nan
thickness[mask_bm!=4] = np.nan
thickness_err[mask_bm!=4] = np.nan

# filter nonsense data (3 sigma)
# max H/Z ratio allowed
# max vals (Amery GL)
e_anomaly = thickness_err - np.nanmean(thickness_err)
ind = np.where((np.isnan(e_anomaly)) | \
               (np.abs(e_anomaly) > 3 * np.nanstd(e_anomaly)) | \
               (freeboard/thickness > 0.33) | \    # TODO Test with this number instead of 0.25!!!!!!!!!!!!!!!
               (freeboard > 250) | \
               (thickness > 2500)
               )
freeboard[ind] = np.nan
thickness[ind] = np.nan
thickness_err[ind] = np.nan

# calculate density and error
density = (1 - freeboard / thickness) * density_water
density_err = (density_water * freeboard / thickness**2) * thickness_err

# smooth fields (this works well with full resolution)
freeboard = convolve(freeboard, Gaussian2DKernel(2), boundary='extend',
                   normalize_kernel=True)
thickness = convolve(thickness, Gaussian2DKernel(2), boundary='extend',
                   normalize_kernel=True)
density = convolve(density, Gaussian2DKernel(2), boundary='extend',
                   normalize_kernel=True)
thickness_err = convolve(thickness_err, Gaussian2DKernel(2), boundary='extend',
                         normalize_kernel=True)
density_err = convolve(density_err, Gaussian2DKernel(2), boundary='extend',
                       normalize_kernel=True)
freeboard[ind] = np.nan
thickness[ind] = np.nan
thickness_err[ind] = np.nan
density[ind] = np.nan
density_err[ind] = np.nan

# interpolate data coords onto density field
freeboard_ = interp(freeboard, x_bm, y_bm, xx, yy, order=1)
freeboard2 = interp(freeboard, x_bm, y_bm, xx, yy, order=0)
freeboard_[np.isnan(freeboard_)] = freeboard2[np.isnan(freeboard_)]

thickness_ = interp(thickness, x_bm, y_bm, xx, yy, order=1)
thickness2 = interp(thickness, x_bm, y_bm, xx, yy, order=0)
thickness_[np.isnan(thickness_)] = thickness2[np.isnan(thickness_)]

thickness_err_ = interp(thickness_err, x_bm, y_bm, xx, yy, order=1)
thickness_err2 = interp(thickness_err, x_bm, y_bm, xx, yy, order=0)
thickness_err_[np.isnan(thickness_err_)] = thickness_err2[np.isnan(thickness_err_)]

density_ = interp(density, x_bm, y_bm, xx, yy, order=1)
density2 = interp(density, x_bm, y_bm, xx, yy, order=0)
density_[np.isnan(density_)] = density2[np.isnan(density_)]

density_err_ = interp(density_err, x_bm, y_bm, xx, yy, order=1)
density_err2 = interp(density_err, x_bm, y_bm, xx, yy, order=0)
density_err_[np.isnan(density_err_)] = density_err2[np.isnan(density_err_)]


if 1: # save
    f = tb.open_file(FILE_OUT, 'w')
    f.create_array('/', 'freeboard', freeboard_)
    f.create_array('/', 'freeboard_full', freeboard)

    f.create_array('/', 'thickness', thickness_)
    f.create_array('/', 'thickness_err', thickness_err_)
    f.create_array('/', 'thickness_full', thickness)
    f.create_array('/', 'thickness_full_err', thickness_err)

    f.create_array('/', 'density', density_)
    f.create_array('/', 'density_err', density_err_)
    f.create_array('/', 'density_full', density)
    f.create_array('/', 'density_full_err', density_err)

    f.create_array('/', 'lon', lon)
    f.create_array('/', 'lat', lat)
    f.create_array('/', 'x_ful', x_bm)
    f.create_array('/', 'y_ful', y_bm)
    f.close()
    print 'file out -> ', FILE_OUT


print '-' * 20
print 'min freeboard:', np.nanmin(freeboard).round(1)
print 'mean freeboard:', np.nanmean(freeboard).round(1)
print 'max freeboard:', np.nanmax(freeboard).round(1)
print ''
print 'min thickness:', np.nanmin(thickness).round(1)
print 'mean thickness:', np.nanmean(thickness).round(1)
print 'max thickness:', np.nanmax(thickness).round(1)
print ''
print 'min density:', np.nanmin(density).round(1)
print 'mean density:', np.nanmean(density).round(1)
print 'max density:', np.nanmax(density).round(1)
print 'std density:', np.nanstd(density).round(1)
print ''
print 'min density_err:', np.nanmin(density_err).round(1)
print 'mean density_err:', np.nanmean(density_err).round(1)
print 'max density_err:', np.nanmax(density_err).round(1)
print 'std density_err:', np.nanstd(density_err).round(1)
print '-' * 20

if 0:
    '''
    plt.figure(facecolor='w')
    plt.imshow(freeboard, origin='lower', interpolation='nearest')
    plt.figure(facecolor='w')
    plt.imshow(thickness, origin='lower', interpolation='nearest')
    '''
    plt.figure(facecolor='w')
    cmap = plt.cm.get_cmap('hot_r', 15)
    plt.imshow(density, origin='lower', interpolation='nearest', cmap=cmap,
               vmin=830,vmax=920)
    plt.title('Column-average density (kg/m^3)')
    plt.colorbar()

    plt.figure(facecolor='w')
    cmap = plt.cm.get_cmap('YlGnBu', 15)
    plt.imshow(density_err, origin='lower', interpolation='nearest', cmap=cmap,
               vmin=1, vmax=170)
    plt.title('Density error (kg/m^3)')
    plt.colorbar()

    plt.show()
