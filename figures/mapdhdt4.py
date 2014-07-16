"""

"""
import sys
import numpy as np
import scipy as sp
import pandas as pd
import tables as tb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altimpy as ap
from mpl_toolkits.basemap import interp

import scipy.ndimage as ni

# CHECK AND EDIT THE 'IF 0/1' IN THE CODE
PLOT = False
DIR = '/Users/fpaolo/data/shelves/' 
FILE_IN = 'all_19920716_20111015_shelf_tide_grids_mts.h5.ice_oce'
FILE_OUT = 'mapdhdt4.h5'
FILE_MSK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'

# for plotting
#LON, LAT = 115.125, -67.125   # Totten, B
#LON, LAT = 260.625, -75.125   # PIG, GL
LON, LAT = 186.375, -79.375  # Ross, with problem at the ends


def as_frame(data, z, y, x):
    try:
        return pd.Panel(data, items=z, major_axis=y, minor_axis=x
                        ).to_frame(filter_observations=False).T
    except:
        print 'already a DataFrame'
        return data


def as_array(data):
    try:
        return data.T.to_panel().values
    except:
        print 'already an Array'
        return data


def write_slabs(fid, var_name, data):
    """Save 3d array into several slabs, for XDMF."""
    g = fid.create_group('/', 'data')
    for i, d in enumerate(data):
        fid.create_array(g, var_name +'_%02d' % i, d)


ap.rcparams()

# read data
print('loading data...')
fin = tb.open_file(DIR + FILE_IN)
data = fin.root.dh_mean_mixed_const_xcal[:]
time = ap.num2year(fin.root.time_xcal[:])
lon = fin.root.lon[:]
lat = fin.root.lat[:]
nz, ny, nx = data.shape

print('done')

if 0: # subset (for testing!)
    region = ap.pig
    data, lon, lat = ap.get_subset(region, data, lon, lat)
    xx, yy = np.meshgrid(lon, lat)
    nt, ny, nx = data.shape                # i,j,k = t,y,x

if 1: # read Mask
    print 'loading mask...'
    f = tb.open_file(FILE_MSK, 'r')
    x_msk = f.root.x[:]
    y_msk = f.root.y[:]
    msk = f.root.mask[:]
    f.close()
    print 'done'

if 1: # filter time
    time, data = ap.time_filt(time, data, from_time=1994, to_time=2013)

dt = ap.year2date(time)

if 1: # split the 3d array
    n_split = 4 
    data = np.vsplit(data, n_split)
    time = np.split(time, n_split)

if 1: # fit linear trend
    print 'fitting trends...'
    dhdt = np.empty((n_split,ny,nx), 'f8')
    dt = np.empty(n_split, 'f8')
    for i in range(n_split):
        dhdt[i] = ap.get_slope(time[i], data[i], robust=False)
        dt[i] = time[i].max()
    time = dt
    del data
    print 'done'

if 1: # gauss filter/interpolate before regridding
    from astropy.convolution import convolve, Gaussian2DKernel
    gauss_kernel = Gaussian2DKernel(1)  # std of 1 pixel
    for i in range(n_split):
        dhdt[i] = convolve(dhdt[i], gauss_kernel, boundary='wrap', normalize_kernel=True)

if 1: # regrid 
    dhdt, lon, lat = ap.regrid2d(dhdt, lon, lat, inc_by=4)

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
    dhdt[:,i,j] = np.nan
    print 'done'

# spherical -> cartesian
lon_nodes, lat_nodes = ap.cell2node(lon, lat)
xx_nodes, yy_nodes = np.meshgrid(lon_nodes, lat_nodes)
xyz = np.column_stack(ap.sph2xyz(xx_nodes.ravel(), yy_nodes.ravel()))

#---------------------------------------------------------------------
# save data 
#---------------------------------------------------------------------

if 1:
    print 'saving data...'
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
    fout.create_array('/', 'dhdt', dhdt)
    write_slabs(fout, 'dhdt', dhdt)
    fout.flush()
    fout.close()
    print 'done'
    print 'out -> ' + DIR + FILE_OUT
fin.close()



