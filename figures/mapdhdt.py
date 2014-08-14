#!/urs/bin/env python

import os
import sys
import numpy as np
import tables as tb
import pyproj as pj
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage as ni
from mpl_toolkits.basemap import interp
from astropy.convolution import convolve, Gaussian2DKernel

import altimpy as ap

print 1

# parameters
#---------------------------------------------------------------------

FILE_DATA = '/Users/fpaolo/data/shelves/h_trendfit.h5'
FILE_IMG = '/Users/fpaolo/data/lima/tiff_90pct/00000-20080319-092059124.tif'
#FILE_IMG = '/Users/fpaolo/data/MOA/moa750_r1_hp1.tif'
FILE_MASK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'
FILE_DEM = '/Users/fpaolo/data/topo/krigged_dem_nsidc.bin'
FILE_GL = '/Users/fpaolo/data/coastline/antarctica_gl_ll.h5'
FILE_COAST = '/Users/fpaolo/data/coastline/moa_coastfile_ll.h5'
FILE_ISL = '/Users/fpaolo/data/coastline/moa_islands_ll.h5'

BBOX_REG = (-131, -60, 51, -58)       # Antarctica

# downsample
step = 20

#---------------------------------------------------------------------

def plot_boundary(m, fname, region=(-180,180,-90,-50), dsample=10, size=0.1,
                    color='k', facecolor='0.8', **kw):
    l, r, b, t = region
    f = tb.open_file(fname)
    d = f.root.data[::dsample]
    lat, lon = d[:,0], d[:,1]
    ii, = np.where((lon >= l) & (lat <= r) & (lat >= b) & (lat <= t))
    x, y = m(lon[ii], lat[ii])
    m.scatter(x, y, s=size, c=color, facecolor=facecolor, lw=0, **kw)
    return m

# data
#---------------------------------------------------------------------

print 2
# read data <- HDF5
f = tb.open_file(FILE_DATA)
time = f.root.time[:]
dhdt = f.root.poly_lasso_rate[:] * 1e2 # m -> cm
lon = f.root.lon[:]
lat = f.root.lat[:]
f.close()

# read Mask <- HDF5
f = tb.open_file(FILE_MASK, 'r')
x_mask = f.root.x[::4]
y_mask = f.root.y[::4]
mask = f.root.mask[::4,::4]
f.close()

# read DEM <- Bin
dem = np.fromfile(FILE_DEM, dtype='<f4').reshape(5601, 5601)[::step,::step]
x_dem = (np.arange(-2800500.0000, 2800500.0000, 1000) + 500)[::step]
y_dem = x_dem[::-1]
xx_dem, yy_dem = np.meshgrid(x_dem, y_dem)

# read IMG <- GeoTIFF, grayscale/rgb
img, georef = ap.read_gtif(FILE_IMG, max_dim=2048.)  # FIXME, return 'img, x, y'?
x_left, y_top, x_right, y_bottom = georef

if 0: # denoise
    img = ni.median_filter(img, 3)

# polar stere coords
x_img = np.linspace(x_left, x_right, img.shape[1])
y_img = np.linspace(y_top, y_bottom, img.shape[0])

# flip raster to match mask array
img = img[::-1,...]
y_img = y_img[::-1]
xx_img, yy_img = np.meshgrid(x_img, y_img)

# bbox of raster img in lon/lat
proj = pj.Proj(proj='stere', lat_ts=-71, lat_0=-90, lon_0=0)
x_left, y_bottom = proj(x_left, y_bottom, inverse=True)
x_right, y_top = proj(x_right, y_top, inverse=True)
bbox_img = (x_left, y_bottom, x_right, y_top)

# process data
#---------------------------------------------------------------------

# convert coords to polar stereographic
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

if 1: # apply mask to data
    print 'applying mask...'
    xx, yy = np.meshgrid(lon, lat)
    xx_polar, yy_polar = ap.ll2xy(xx, yy, units='m')
    mask_dhdt = interp(mask[::-1,:], x_mask, y_mask[::-1], xx_polar, yy_polar, order=0) # nn
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

# colormaps, background, foreground
#---------------------------------------------------------------------

# figure
fig = plt.figure(figsize=(10,9), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# Custom colormap for ice
colors = plt.cm.gray(np.linspace(0.2, 0.9, 256))
cmap_ice = ap.make_cmap(colors)

# Custom colormap for data, with especific bounds for normalization.
'''
# logarithmic
bounds = np.r_[
               np.arange(-35, -5, 10),
               np.arange(-5, 5, 1),
               np.arange(5, 40, 5),
              ]
colors = np.loadtxt('/Users/fpaolo/data/mpl/NCV_blu_red.rgb')[::-1]
cmap_data = ap.make_cmap(colors)
norm = mpl.colors.SymLogNorm(5)

'''
# linear
bounds = np.r_[
               np.arange(-35, -5, 5),
               [-5, -2],
               np.arange(-2, 21, 1),
              ]
colors = np.loadtxt('/Users/fpaolo/data/mpl/NCV_blu_red.rgb')[::-1]
cmap_data = ap.make_cmap(colors)
cmap_data = ap.shift_cmap(cmap_data, midpoint=bounds)
norm = None

print 'bounds N =', len(bounds), bounds

if 1:
    # interpolate MOA coords onto Mask
    mask_img = interp(mask[::-1,:], x_mask, y_mask[::-1], xx_img, yy_img, order=0)

    # interpolate MOA coords onto DEM
    dem_img = interp(dem[::-1,:], x_dem, y_dem[::-1], xx_img, yy_img, order=0)

    # interpolate dh/dt onto Mask
    del mask_dhdt
    mask_dhdt = interp(mask[::-1,:], x_mask, y_mask[::-1], xx_dh, yy_dh, order=0)

    # create shaded image
    '''
    img[img<60] = 60  # stretch img image
    img[img>150] = 150
    '''
    img = ap.shade2(img, cmap=plt.cm.jet,
                    intensity=ap.hillshade(dem_img, scale=20, azdeg=165, altdeg=45))
    #img = ap.shade(dem_img, cmap=cmap_ice, scale=0.02, azdeg=165, altdeg=45)
    #img = ap.shade2(dem_img, cmap=cmap_ice, scale=0.03, azdeg=165, altdeg=45)

    # FIXME
    # convert int to float: [0,255] to [0,1]
    if img.max() > 1:
        img = img / 255.

    alpha = np.ones(img.shape[:2], 'uint8')
    img2 = img.copy()
    alpha2 = alpha.copy()

    # mask out everything but land and connected islands in MOA 1
    #img[(mask_img!=1) & (mask_img!=2),...] = np.nan
    alpha[(mask_img!=1) & (mask_img!=2)] = 0

    # mask out everything but ice shelves in MOA 2
    #img2[mask_img!=4,...] = np.nan
    alpha2[mask_img!=4] = 0

    # mask out ocean in dh/dt
    dhdt[mask_dhdt==0] = np.nan

    img = np.dstack((img, alpha))
    img2 = np.dstack((img2, alpha2))

    print img
    print img.dtype
    print img.shape
    #exit()

# main plot
#---------------------------------------------------------------------

m = ap.make_proj_stere(BBOX_REG)

# current region is smaller than image extent
m, _, _, _ = ap.plot_img_subreg(m, x_img, y_img, img2, bbox_img, zorder=1)

# Generate 2d grid with coords and transform stere
xx, yy = np.meshgrid(lon, lat)
xx, yy = m(xx, yy)

# set under and over for colors
dhdt[dhdt<bounds[0]] = bounds[0]
dhdt[dhdt>bounds[-1]] = bounds[-1]

c1 = m.contourf(xx, yy, dhdt, bounds, cmap=cmap_data, norm=norm, zorder=2)

cax = fig.add_axes([0.045, 0.92, 0.3, 0.02])
plt.colorbar(c1, cax=cax, orientation='horizontal', spacing='proportional',
             ticks=[bounds[0], 0, bounds[-1]], drawedges=False)

ax = fig.add_axes([0, 0, 1, 1])
m, _, _, _ = ap.plot_img_subreg(m, x_img, y_img, img, bbox_img, cmap=None, zorder=3)

m = plot_boundary(m, FILE_GL, size=0.1, color='.7', facecolor='0.1', zorder=4)
m = plot_boundary(m, FILE_COAST, size=0.1, color='.9', facecolor='0.1', zorder=5)

plt.show()
exit()

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

axs = plt.axes([0.075,0.082,.4,.11])

'''
ap.adjust_spines(axs, [])
axs.spines['left'].set_position(('outward', 4))
axs.spines['bottom'].set_position(('outward', 4))
axs.spines['left'].set_linewidth(1.5)
axs.spines['bottom'].set_linewidth(1.5)

axs.tick_params(axis='both', direction='out', length=4, width=2, pad=-4,
                labelsize=11)
axs.yaxis.set_ticks_position('left')
axs.xaxis.set_ticks_position('bottom')
'''
ap.adjust_spines(axs, ['left', 'bottom'], pad=15)
axs.tick_params(axis='both', direction='out', length=6, width=1,
                labelsize=11)

plt.xticks([1994, 1997, 2000, 2003, 2006, 2009, 2012])
plt.xticks(rotation=30)
plt.yticks([-28, 0, 28])
plt.ylim(-28,28)
plt.xlim(1994, 2012)
plt.ylabel('cm')

plt.plot(t, h_wais, 'sr', markersize=3, clip_on=False, alpha=0.5, zorder=3)
plt.plot(t, h_eais, 'sb', markersize=3, clip_on=False, alpha=0.5, zorder=3)
plt.plot(t, p_wais, c='r', linewidth=2, clip_on=False, zorder=2)
plt.plot(t, p_eais, c='b', linewidth=2, clip_on=False, zorder=2)
axs.plot(t, np.zeros(len(t)), '-', c='0.5', linewidth=0.75, zorder=1)

plt.text(0.27, 0.1,'EAIS',
         fontsize=12,
         color='b',
         horizontalalignment='center',
         verticalalignment='center',
         transform=axs.transAxes)

plt.text(0.75, 0.1,'WAIS',
         fontsize=12,
         color='r',
         horizontalalignment='center',
         verticalalignment='center',
         transform=axs.transAxes)

# colorbar and text
#---------------------------------------------------------------------

'''
legend = '1994-2012\nElevation-change rate (cm/yr)'
ticklabels = labels # '< %g' % -MAX_DHDT, 'cm/yr', '> %g' % MAX_DHDT
ticks = position # -MAX_DHDT, 0, MAX_DHDT 
colorlim = ticks[0], ticks[-1]
colorexp = 1.0
inches = 6.4, 3.6
extent = (-121.1, -113.8), (32.1, 35.5)
ylim = 150.0
xlim = ylim * inches[0] / inches[1]
axis = -xlim, xlim, -ylim, ylim
'''

if 0:
    #cmap.set_over('b')
    #cmap.set_under('r')
    rect = 0.06, 0.9, 0.24, 0.02  # upper-left
    #rect = 0.55, 0.5, 0.26, 0.02   # center
    #rect = 0.695, 0.95, 0.25, 0.018  # upper-right
    ap.colorbar(fig, cmap_data, colorlim, legend, rect, ticks, ticklabels,
                boxalpha=1, size=14, weight='normal', color='k', edgecolor='w',
                edgealpha=.5, edgewidth=1, shifttitle=1.6, shiftlabel=-0.05)

if 0:
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
    plt.savefig('Fig1_dhdt_map_v2.png', dpi=150, bbox_inches='tight')
    print 'done.'

plt.show()
