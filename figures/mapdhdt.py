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

import numba

# parameters
#---------------------------------------------------------------------

FILE_IN = '/Users/fpaolo/data/shelves/h_trendfit.h5'

MOAFILE = '/Users/fpaolo/data/MOA/moa750_r1_hp1.tif'
#MOAFILE = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.tif'
MASKFILE = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'
DEMFILE = '/Users/fpaolo/data/topo/krigged_dem_nsidc.bin'
GLFILE = '/Users/fpaolo/data/coastline/antarctica_gl_ll.h5'
COASTFILE = '/Users/fpaolo/data/coastline/moa_coastfile_ll.h5'
ISLFILE = '/Users/fpaolo/data/coastline/moa_islands_ll.h5'

MAX_DHDT = 10  # (cm) for color range
FROM_YEAR, TO_YEAR = 1992, 2013
#BBOX_REG = (-82, -76.2, -26, -79.5)  # FRIS
#BBOX_REG = (-156, -76, 154, -81.2)   # ROSS
#BBOX_REG = (68, -74.5, 70, -67.5)    # AMERY
BBOX_REG = (-131, -60, 51, -58)       # Antarctica

# downsample
step = 10 

#ap.rcparams()

cmap = ap.get_cmap('r2b', n=21)
#cmap = ap.get_cmap('bwr', n=11, invert=True)
#cmap = plt.cm.get_cmap('bwr_r', 11)
#cmap = plt.cm.get_cmap('seismic_r', 11)
legend = 'Elevation-change rate 1994-2012'
ticklabels = '< %g' % -MAX_DHDT, '0', '> %g cm/yr' % MAX_DHDT
ticks = -MAX_DHDT, 0, MAX_DHDT 
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

# input
#---------------------------------------------------------------------

# read data
f = tb.open_file(FILE_IN)
time = f.root.time[:]
dhdt = f.root.poly_lasso_rate[:] * 1e2 # m -> cm
#dhdt = f.root.line_lstsq_rate[:] * 1e2
lon = f.root.lon[:]
lat = f.root.lat[:]
f.close()

# read Mask
f = tb.open_file(MASKFILE, 'r')
x_msk = f.root.x[::4]
y_msk = f.root.y[::4]
msk = f.root.mask[::4,::4]
f.close()

# read DEM
dem = np.fromfile(DEMFILE, dtype='<f4').reshape(5601, 5601)[::step,::step]
x_dem = (np.arange(-2800500.0000, 2800500.0000, 1000) + 500)[::step]
y_dem = x_dem[::-1]
xx_dem, yy_dem = np.meshgrid(x_dem, y_dem)

# read MOA
m = ap.make_proj_stere(BBOX_REG)
x_moa, y_moa, moa, bbox_moa = ap.get_gtif(MOAFILE, units='m')
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
    dhdt = np.c_[dhdt, dhdt[:,-1]]
    lon = np.r_[lon, lon[-1]+dx]
    dhdt = np.c_[dhdt, dhdt[:,-1]]
    lon = np.r_[lon, lon[-1]+dx]

if 1: # smooth and interpolate before regridding
    gauss_kernel = Gaussian2DKernel(1)  # std of 1 pixel
    dhdt = convolve(dhdt, gauss_kernel, boundary='wrap', normalize_kernel=True)

if 1: # regrid 
    lon, lat, dhdt = ap.regrid(lon, lat, dhdt, inc_by=4)

if 0: # smooth after regridding 
    dhdt = np.c_[dhdt[:,:10], dhdt, dhdt[:,-10:]]
    '''
    ind = np.where(np.isnan(dhdt))
    dhdt[ind] = 0
    dhdt = ni.gaussian_filter(dhdt, 2)
    dhdt[ind] = np.nan
    '''
    dhdt = ap.median_filt(dhdt, 9, min_pixels=2)
    dhdt = dhdt[:, 10:-10]

if 1: # apply mask to data
    print 'applying mask...'
    xx, yy = np.meshgrid(lon, lat)
    xx_polar, yy_polar = ap.ll2xy(xx, yy, units='m')
    mask_dhdt = interp(msk[::-1,:], x_msk, y_msk[::-1], xx_polar, yy_polar, order=0) # nn
    mask_dhdt = mask_dhdt.astype('f8')
    # remove everything but ice shelves
    mask_dhdt[mask_dhdt!=4] = np.nan    
    # remove grid cells outside satellite limit
    i, = np.where(lat < -81.6)          
    mask_dhdt[i,:] = np.nan
    # mask out data
    i, j = np.where(np.isnan(mask_dhdt))
    dhdt[i,j] = np.nan
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
    mask_moa = interp(msk[::-1,:], x_msk, y_msk[::-1], xx_moa, yy_moa, order=0)

    # interpolate MOA onto DEM
    dem_moa = interp(dem[::-1,:], x_dem, y_dem[::-1], xx_moa, yy_moa, order=0)

    # interpolate dh/dt onto Mask
    del mask_dhdt
    mask_dhdt = interp(msk[::-1,:], x_msk, y_msk[::-1], xx_dh, yy_dh, order=0)

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
    dhdt[mask_dhdt==0] = np.nan

# main plot
#---------------------------------------------------------------------

m, _, _, _ = ap.plot_moa_subreg(m, x_moa, y_moa, moa2[...,0], bbox_moa, zorder=1)

m = ap.plot_grid_proj(m, lon, lat, dhdt, cmap=cmap, alpha=1, vmin=-MAX_DHDT,
                      vmax=MAX_DHDT, zorder=2)

m, _, _, _ = ap.plot_moa_subreg(m, x_moa, y_moa, moa[...,0], bbox_moa, zorder=3)

#m = plot_boundary(m, GLFILE, size=0.1, color='k', facecolor='0.5', zorder=4)
#m = plot_boundary(m, COASTFILE, size=0.5, color='k', facecolor='0.1', zorder=5)
#mm = m.drawmeridians(np.arange(-80.,-20.,7), labels=[0,1,0,0], color='0.6')
#pp = m.drawparallels(np.arange(-90.,-81.5,8), labels=[0,0,1,0], color='0.9', 
#                     zorder=5)

# inset plot
#---------------------------------------------------------------------

t = np.loadtxt('Time.txt')
h_wais = ap.referenced(np.loadtxt('West.txt'), to='mean') # already in cm!
h_eais = ap.referenced(np.loadtxt('East.txt'), to='mean')

p_wais = ap.lasso_cv(t, h_wais, cv=10, max_deg=3)
p_eais = ap.lasso_cv(t, h_eais, cv=10, max_deg=3)

axs = plt.axes([0.07,0.08,.4,.1])

ap.adjust_spines(axs, [])
axs.spines['left'].set_position(('outward', 4))
axs.spines['bottom'].set_position(('outward', 4))
axs.spines['left'].set_linewidth(1.5)
axs.spines['bottom'].set_linewidth(1.5)

axs.tick_params(axis='both', direction='out', length=4, width=2, pad=-4,
                labelsize=11)
axs.yaxis.set_ticks_position('left')
axs.xaxis.set_ticks_position('bottom')
plt.xticks([1994, 1997, 2000, 2003, 2006, 2009, 2012])
plt.xticks(rotation=30)
plt.yticks([-28, 0, 28])
plt.ylim(-28,28)
plt.xlim(1994, 2012)
plt.ylabel('cm')

plt.plot(t, h_wais, 'sr', markersize=3.5, clip_on=False, alpha=0.6, zorder=3)
plt.plot(t, h_eais, 'sb', markersize=3.5, clip_on=False, alpha=0.6, zorder=3)
plt.plot(t, p_wais, c='r', linewidth=1.5, zorder=2)
plt.plot(t, p_eais, c='b', linewidth=1.5, zorder=2)
axs.plot(t, np.zeros(len(t)), '-', c='0.5', linewidth=0.75, zorder=1)
ap.intitle('WAIS', ax=axs, pad=-.8, loc=1)
ap.intitle('EAIS', ax=axs, pad=-.8, loc=2)

#plt.text(0.5,0.5, 'axes([0.2,0.2,.3,.3])',ha='center',va='center',size=16,alpha=.5)

#---------------------------------------------------------------------

# colorbar and text
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

if 0:
    # try ap.savefig and ap.img2pdf !!!
    print 'saving figure...'
    plt.savefig('Fig1_map_v2.png', dpi=150, bbox_inches='tight')
    print 'done.'

plt.show()
