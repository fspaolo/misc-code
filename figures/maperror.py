#!/urs/bin/env python

import os
import sys
import numpy as np
import tables as tb
import matplotlib as mpl
import matplotlib.pyplot as plt
import altimpy as ap
from scipy import ndimage as ni
from mpl_toolkits.basemap import interp
from astropy.convolution import convolve, Gaussian2DKernel

# parameters
#---------------------------------------------------------------------

FILE_DATA = '/Users/fpaolo/data/shelves/h_trendfit.h5'
FILE_MOA = '/Users/fpaolo/data/MOA/moa750_r1_hp1.tif'
#FILE_MOA = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.tif'
FILE_MASK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'
FILE_DEM = '/Users/fpaolo/data/topo/krigged_dem_nsidc.bin'
FILE_GL = '/Users/fpaolo/data/coastline/antarctica_gl_ll.h5'
FILE_COAST = '/Users/fpaolo/data/coastline/moa_coastfile_ll.h5'
FILE_ISL = '/Users/fpaolo/data/coastline/moa_islands_ll.h5'

MIN_ERROR = .3  # (cm) for color range
MAX_ERROR = 3 
#BBOX_REG = (-82, -76.2, -26, -79.5)  # FRIS
#BBOX_REG = (-156, -76, 154, -81.2)   # ROSS
#BBOX_REG = (68, -74.5, 70, -67.5)    # AMERY
BBOX_REG = (-131, -60, 51, -58)       # Antarctica

# downsample
step = 2

cmap = ap.get_cmap('w2r', n=20)
legend = 'Rate error'
ticklabels = '< %g' % MIN_ERROR, 'cm/yr', '> %g' % MAX_ERROR
ticks = MIN_ERROR, (MAX_ERROR-MIN_ERROR)/1.6, MAX_ERROR 
colorlim = ticks[0], ticks[-1]
colorexp = 1.0
inches = 6.4, 3.6
extent = (-121.1, -113.8), (32.1, 35.5)
ylim = 150.0
xlim = ylim * inches[0] / inches[1]
axis = -xlim, xlim, -ylim, ylim

#---------------------------------------------------------------------

def plot_boundary(m, fname, region=(-180,180,-90,-50), res=10, size=0.1,
                    color='k', facecolor='0.8', **kw):
    l, r, b, t = region
    f = tb.open_file(fname)
    d = f.root.data[::res]
    lat, lon = d[:,0], d[:,1]
    ii, = np.where((lon >= l) & (lat <= r) & (lat >= b) & (lat <= t))
    x, y = m(lon[ii], lat[ii])
    m.scatter(x, y, s=size, c=color, facecolor=facecolor, lw = 0, **kw)
    return m

# data
#---------------------------------------------------------------------

# read data
f = tb.open_file(FILE_DATA)
time = f.root.time[:]
error = f.root.poly_lasso_rate_error[:] * 3 * 1e2 # m -> cm
#error = f.root.line_lstsq_rate[:] * 1e2
lon = f.root.lon[:]
lat = f.root.lat[:]
f.close()

# read Mask
f = tb.open_file(FILE_MASK, 'r')
x_mask = f.root.x[::step]
y_mask = f.root.y[::step]
mask = f.root.mask[::step,::step]
f.close()

# read DEM
dem = np.fromfile(FILE_DEM, dtype='<f4').reshape(5601, 5601)[::step,::step]
x_dem = (np.arange(-2800500.0000, 2800500.0000, 1000) + 500)[::step]
y_dem = x_dem[::-1]
xx_dem, yy_dem = np.meshgrid(x_dem, y_dem)

# read MOA
m = ap.make_proj_stere(BBOX_REG)
x_moa, y_moa, moa, bbox_moa = ap.get_gtif(FILE_MOA, units='m')
x_moa, y_moa, moa = x_moa[::step], y_moa[::step], moa[::step,::step]
xx_moa, yy_moa = np.meshgrid(x_moa, y_moa)
moa = ni.median_filter(moa, 3)   # denoise!!!

# process data
#---------------------------------------------------------------------

# convert to polar stereographic
xx_dh, yy_dh = np.meshgrid(lon, lat)
xx_dh, yy_dh = ap.ll2xy(xx_dh, yy_dh, units='m')

if 1: # add 2 slices for discontinuity at 360-0 BOUNDARY!!!
    dx = lon[1] - lon[0]
    error = np.c_[error, error[:,-1]]
    lon = np.r_[lon, lon[-1]+dx]
    error = np.c_[error, error[:,-1]]
    lon = np.r_[lon, lon[-1]+dx]

if 1: # smooth and interpolate before regridding
    gauss_kernel = Gaussian2DKernel(1)  # std of 1 pixel
    error = convolve(error, gauss_kernel, boundary='wrap', normalize_kernel=True)

if 1: # regrid 
    lon, lat, error = ap.regrid(lon, lat, error, inc_by=4)

if 0: # smooth after regridding 
    error = np.c_[error[:,:10], error, error[:,-10:]]
    '''
    ind = np.where(np.isnan(error))
    error[ind] = 0
    error = ni.gaussian_filter(error, 2)
    error[ind] = np.nan
    '''
    error = ap.median_filt(error, 9, min_pixels=2)
    error = error[:, 10:-10]

if 1: # apply mask to data
    print 'applying mask...'
    xx, yy = np.meshgrid(lon, lat)
    xx_polar, yy_polar = ap.ll2xy(xx, yy, units='m')
    mask_error = interp(mask[::-1,:], x_mask, y_mask[::-1], xx_polar, yy_polar, order=0) # nn
    mask_error = mask_error.astype('f8')
    # remove everything but ice shelves
    mask_error[mask_error!=4] = np.nan    
    # remove grid cells outside satellite limit
    i, = np.where(lat < -81.6)          
    mask_error[i,:] = np.nan
    # mask out data
    i, j = np.where(np.isnan(mask_error))
    error[i,j] = np.nan
    print 'done'

# process background/forground 
#---------------------------------------------------------------------

# figure
fig = plt.figure(figsize=(10,9), frameon=True)
fig.patch.set_facecolor('white')
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

if 1:
    # interpolate MOA onto Mask
    mask_moa = interp(mask[::-1,:], x_mask, y_mask[::-1], xx_moa, yy_moa, order=0)

    # interpolate MOA onto DEM
    dem_moa = interp(dem[::-1,:], x_dem, y_dem[::-1], xx_moa, yy_moa, order=0)

    # interpolate dh/dt onto Mask
    del mask_error
    mask_error = interp(mask[::-1,:], x_mask, y_mask[::-1], xx_dh, yy_dh, order=0)

    # create shaded image. scale up -> vert. exag. down
    moa[moa<60] = 60
    moa[moa>150] = 150
    moa = ap.shade(moa, cmap=plt.cm.gray, 
                   intensity=ap.hillshade(dem_moa, scale=15, azdeg=165, altdeg=45))
    moa2 = moa.copy()

    # mask out everything but land and connected islands in MOA 1
    moa[(mask_moa!=1) & (mask_moa!=2)] = 10
    moa = np.ma.masked_values(moa, 10)

    # mask out everything but ice shelves in MOA 2
    moa2[mask_moa!=4] = 10
    moa2 = np.ma.masked_values(moa2, 10)

    # mask out ocean in dh/dt
    error[mask_error==0] = np.nan

# main plot
#---------------------------------------------------------------------

m, _, _, _ = ap.plot_moa_subreg(m, x_moa, y_moa, moa2[...,0], bbox_moa, zorder=1)

m = ap.plot_grid_proj(m, lon, lat, error, cmap=cmap, alpha=1, vmin=MIN_ERROR,
                      vmax=MAX_ERROR, zorder=2)

m, _, _, _ = ap.plot_moa_subreg(m, x_moa, y_moa, moa[...,0], bbox_moa, zorder=3)

#m = plot_boundary(m, FILE_GL, size=0.1, color='k', facecolor='0.5', zorder=4)
#m = plot_boundary(m, FILE_COAST, size=0.5, color='k', facecolor='0.1', zorder=5)
#mm = m.drawmeridians(np.arange(-80.,-20.,7), labels=[0,1,0,0], color='0.6')
#pp = m.drawparallels(np.arange(-90.,-81.5,8), labels=[0,0,1,0], color='0.9', 
#                     zorder=5)

# colorbar and text
#---------------------------------------------------------------------

#cmap.set_over('b')
#cmap.set_under('r')
rect = 0.06, 0.9, 0.24, 0.02  # upper-left
#rect = 0.55, 0.5, 0.26, 0.025   # center
ap.colorbar(fig, cmap, colorlim, legend, rect, ticks, ticklabels, boxalpha=1,
            size=14, weight='normal', color='k', edgecolor='w', edgealpha=.5,
            edgewidth=1)

px, py = [700, 1200], [3700, 3700]
ap.length_scale(ax, px, py, label='500 km\n\n', style='k-', linewidth=1.5,
                color='k', weight='normal', zorder=10)
# automatic: label='%s km\n\n'

#ap.intitle('Filchner-Ronne Ice Shelf', loc=1)
#ap.intitle('dh/dt 1992-2012', loc=3)
#leg = fig.add_axes([0, 0, 1, 1])
#leg.set_axis_off()

if 1:
    # try ap.savefig and ap.img2pdf !!!
    print 'saving figure...'
    plt.savefig('Sup3_error_map_v1.png', dpi=150, bbox_inches='tight')
    print 'done.'

plt.show()
