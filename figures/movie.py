#!/urs/bin/env python

import os
import sys
import numpy as np
import tables as tb
import pandas as pd
import pyproj as pj
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage as ni
from mpl_toolkits.basemap import interp
from astropy.convolution import convolve, Gaussian2DKernel

import altimpy as ap

# parameters
#---------------------------------------------------------------------

FILE_DATA = '/Users/fpaolo/data/shelves/h_trendfit.h5'
FILE_OUT = '/Users/fpaolo/data/shelves/dhdt_poly_movie.h5'

FILE_IMG = '/Users/fpaolo/data/lima/tiff_90pct/00000-20080319-092059124.tif'
#FILE_IMG = '/Users/fpaolo/data/MOA/moa750_r1_hp1.tif'
FILE_MASK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'
FILE_DEM = '/Users/fpaolo/data/topo/krigged_dem_nsidc.bin'
FILE_GL = '/Users/fpaolo/data/coastline/antarctica_gl_ll.h5'
FILE_COAST = '/Users/fpaolo/data/coastline/moa_coastfile_ll.h5'
FILE_ISL = '/Users/fpaolo/data/coastline/moa_islands_ll.h5'

# downsample
step = 20

#---------------------------------------------------------------------

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

# data
#---------------------------------------------------------------------

print 'loading...'

# read data <- HDF5
f = tb.open_file(FILE_DATA)
time = f.root.time[:]
#data = f.root.poly_lasso[:] * 1e2 # m -> cm
data = f.root.dpoly_lasso[:] * 1e2 # m -> cm
lon = f.root.lon[:]
lat = f.root.lat[:]
f.close()

# read Mask <- HDF5
f = tb.open_file(FILE_MASK, 'r')
x_mask = f.root.x[::4]
y_mask = f.root.y[::4]
mask = f.root.mask[::4,::4]
f.close()

'''
# read DEM <- Bin
dem = np.fromfile(FILE_DEM, dtype='<f4').reshape(5601, 5601)[::step,::step]
x_dem = (np.arange(-2800500.0000, 2800500.0000, 1000) + 500)[::step]
y_dem = x_dem[::-1]
xx_dem, yy_dem = np.meshgrid(x_dem, y_dem)

# read IMG <- GeoTIFF, grayscale/rgb
img, georef = ap.read_gtif(FILE_IMG, max_dim=2048.)
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
'''

# process data
#---------------------------------------------------------------------

print 'processing...'

if 0: # DO NOT USE THIS FOR dh/dt
    data = as_frame(data, time, lat, lon)
    data = data.apply(ap.referenced, to='first', raw=True)
    data = as_array(data)

if 1: # add 2 slices for discontinuity at 360-0 BOUNDARY!!!
    data_ = np.zeros((data.shape[0], data.shape[1], data.shape[2]+2), 'f8')
    data_[...,:-2] = data
    data_[...,-2] = data[...,-1]
    data_[...,-1] = data[...,-1]
    dx = lon[1] - lon[0]
    lon = np.r_[lon, lon[-1]+dx, lon[-1]+2*dx]
    data = data_

if 1: # smooth and interpolate before regridding
    gauss_kernel = Gaussian2DKernel(1)  # std of 1 pixel
    for k in range(data.shape[0]):
        data[k] = convolve(data[k], gauss_kernel, boundary='wrap', normalize_kernel=True)

if 1: # regrid 
    data, lon, lat = ap.regrid2d(data, lon, lat, inc_by=4)

if 1: # apply mask to data
    print 'applying mask...'
    xx, yy = np.meshgrid(lon, lat)
    xx_polar, yy_polar = ap.ll2xy(xx, yy, units='m')
    mask_data = interp(mask[::-1,:], x_mask, y_mask[::-1], xx_polar, yy_polar, order=0) # nn
    mask_data = mask_data.astype('f8')
    # remove everything but ice shelves
    mask_data[mask_data!=4] = np.nan    
    # remove grid cells outside satellite limit
    i, = np.where(lat < -81.6)          
    mask_data[i,:] = np.nan
    # mask out data
    i, j = np.where(np.isnan(mask_data))
    data[:,i,j] = np.nan
    print 'done'

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
    fout = tb.open_file(FILE_OUT, 'w')
    # required
    #---------------------------------------------
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
    # optional
    #---------------------------------------------
    '''
    write_slabs(fout, 'poly_lasso', data)
    fout.create_array('/', 'poly_lasso', data)
    '''
    write_slabs(fout, 'dpoly_lasso', data)
    fout.create_array('/', 'dpoly_lasso', data)

    fout.flush()
    fout.close()

    print 'out -> ', FILE_OUT
