"""
Calculate the area of each grid cell by counting the number of 1 x 1 km mask
cells that fall into the data cells.

Notes
-----
The full grid takes about 7.25 hours

"""

import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt
import altimpy as ap

DIR = '/Users/fpaolo/data/shelves/' 
FILE_MSK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'
FILE_IN = 'all_19920716_20111015_shelf_tide_grids_mts.h5.ice_oce'
FILE_OUT = 'area_grid_cells_new.h5'  # FIXME <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
SAVE = True

print 'loading data...'
fin = tb.openFile(DIR + FILE_IN)
lon = fin.root.lon[:]
lat = fin.root.lat[:]
data = fin.root.dh_mean_mixed_const_xcal[:]

if 1: # increment the resolution of the grid
    inc = 5
    lon = np.linspace(lon[0], lon[-1], lon.shape[0] * inc)
    lat = np.linspace(lat[0], lat[-1], lat.shape[0] * inc)

if not SAVE: # subset (for testing!)
    data, _ = np.meshgrid(lon, lat)
    region = ap.crosson
    data, lon, lat = ap.get_subset(region, data, lon, lat)

# load mask
print 'loading mask...'
f = tb.open_file(FILE_MSK, 'r')
x_msk = f.root.x[:]
y_msk = f.root.y[:]
msk = f.root.mask[:]
f.close()
print 'done'

# 2d -> 1d
x_msk, y_msk = np.meshgrid(x_msk, y_msk)    # 1d -> 2d
x_msk, y_msk = x_msk.ravel(), y_msk.ravel() # 2d -> 1d
msk = msk.ravel()

# x/y -> lon/lat
lon_msk, lat_msk = ap.xy2ll(x_msk, y_msk, units='m')
lon_msk = ap.lon_180_360(lon_msk)
del x_msk, y_msk

lon_nodes, lat_nodes = ap.cell2node(lon, lat)
area = np.full((lat.shape[0], lon.shape[0]), np.nan)
cells_idx = []

# count number of mask cells falling into each data cell
print 'calculating grid-cell area...'
for i in xrange(len(lat)):
    for j in xrange(len(lon)):
        i_cells, = np.where((lon_nodes[j] <= lon_msk) & \
                            (lon_msk <= lon_nodes[j+1]) & \
                            (lat_nodes[i] <= lat_msk) & \
                            (lat_msk <= lat_nodes[i+1]) & \
                            (msk == 4))  # 4 = ice shelf

        # each mask cell is 1 km**2
        area[i,j] = len(i_cells)
        cells_idx = np.append(cells_idx, i_cells)

cells_idx = [int(i) for i in cells_idx]

print 'done'
ind = ap.where_isnan('crosson', lon, lat)
area[ind] = np.nan
print 'Total area (km2):', np.nansum(area)

if SAVE:
    f = tb.open_file(FILE_OUT, 'a')
    '''
    f.create_array('/', 'area', area)
    f.create_array('/', 'lon', lon)
    f.create_array('/', 'lat', lat)
    '''
    f.create_array('/', 'area_5', area)
    f.create_array('/', 'lon_5', lon)
    f.create_array('/', 'lat_5', lat)
    f.flush()
    f.close()

if 1:
    plt.figure()
    '''
    plt.imshow(data[0], origin='lower', interpolation='nearest', 
                extent=(lon_nodes[0], lon_nodes[-1], lat_nodes[0], lat_nodes[-1]))
    '''
    plt.plot(lon_msk[cells_idx], lat_msk[cells_idx], 'xk')
    plt.figure()
    plt.imshow(area, origin='lower', interpolation='nearest')
    plt.show()
    exit()

print 'out ->', FILE_OUT
