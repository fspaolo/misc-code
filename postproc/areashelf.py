"""
calculate the area of individual ice shelves by using the
grid-cell-area mask.

"""
import sys
import numpy as np
import pandas as pd
import tables as tb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altimpy as ap
from scipy.signal import detrend
from patsy import dmatrix
from sklearn.linear_model import LassoCV
from mpl_toolkits.basemap import interp

SAVE = True
#FILE_IN = 'h_raw4.h5'
FILE_OUT = 'ice_shelf_area4.csv'
DIR = '/Users/fpaolo/data/shelves/' 
FILE_MSK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'
FILE_IN = 'all_19920716_20111015_shelf_tide_grids_mts.h5.ice_oce'


shelves = [
    # individual shelves
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
    # regions
    ap.queenmaud,
    ap.fris,
    ap.larsen,
    ap.belling,
    ap.amundsen,
    ap.ris,
    ap.tottmosc,
    ap.westshac,
    None,          # empty on purpose 
    ap.wais,
    ap.ais,
    ]

names = [
    'Fimbul_W',
    'Fimbul_E',
    'Lazarev',
    'Amery',
    'West',
    'Shackleton',
    'Totten',
    'Moscow',
    'Ross_E',
    'Ross_W',
    'Sulzberger',
    'Getz',
    'Dotson',
    'Crosson',
    'Thwaites',
    'Pine_Island',
    'Abbot',
    'Stange',
    'Bach',
    'Wilkins',
    'George_S',
    'George_N',
    'Larsen_B',
    'Larsen_C',
    'Larsen_D',
    'Ronne',
    'Filchner',
    'Brunt',
    'Riiser',
    'Queen_Maud',
    'FRIS',
    'Eastern_AP',
    'Bellingshausen',
    'Amundsen',
    'RIS',
    'Totten_Moscow',
    'West_Shack.',
    'East_Antarctica',
    'West_Antarctica',
    'All_Antarctica',
    ]

print 'loading data...'
fin = tb.openFile(DIR + FILE_IN)
try:
    time = ap.num2year(fin.root.time[:])
except:
    time = fin.root.time[:]
dt = ap.year2date(time)
lon = fin.root.lon[:]
lat = fin.root.lat[:]
data = fin.root.dh_mean_mixed_const_xcal[:]
nz, ny, nx = data.shape


if 0: # subset (for testing!)
    region = ap.larsenc
    data, lon, lat = ap.get_subset(region, data, lon, lat)
    nt, ny, nx = data.shape                # i,j,k = t,y,x

if 1: # load mask
    print 'loading mask...'
    f = tb.open_file(FILE_MSK, 'r')
    x_msk = f.root.x[:]
    y_msk = f.root.y[:]
    msk = f.root.mask[:]
    f.close()
    print 'done'

if 1: # 2d -> 1d, x/y -> lon/lat
    x_msk, y_msk = np.meshgrid(x_msk, y_msk)    # 1d -> 2d
    x_msk, y_msk = x_msk.ravel(), y_msk.ravel() # 2d -> 1d
    msk = msk.ravel()
    lon_msk, lat_msk = ap.xy2ll(x_msk, y_msk, units='m')
    lon_msk = ap.lon_180_360(lon_msk)
    del x_msk, y_msk

lon_nodes, lat_nodes = ap.cell2node(lon, lat)
area = np.full(data[0].shape, np.nan)
cells_idx = []

# count number of mask cells falling into each data cell
for i in xrange(len(lat)):
    for j in xrange(len(lon)):
        i_cells, = np.where((lon_nodes[j] <= lon_msk) & \
                            (lon_msk <= lon_nodes[j+1]) & \
                            (lat_nodes[i] <= lat_msk) & \
                            (lat_msk <= lat_nodes[i+1]) & \
                            (msk == 4))  # 4 = ice shelf
        area[i,j] = len(i_cells)         # each mask cell is 1 km**2
        cells_idx = np.append(cells_idx, i_cells)

cells_idx = [int(i) for i in cells_idx]

print area
print 'Total area (km2):', area.sum()

if SAVE:
    f = tb.open_file('area_grid_cells.h5', 'w')
    f.create_array('/', 'area', area)
    f.create_array('/', 'lon', lon)
    f.create_array('/', 'lat', lat)
    f.flush()
    f.close()

if 0:
    plt.figure()
    plt.imshow(data[0], origin='lower', interpolation='nearest', 
                extent=(lon_nodes[0], lon_nodes[-1], lat_nodes[0], lat_nodes[-1]))
    plt.plot(lon_msk[cells], lat_msk[cells], 'xk')
    plt.figure()
    plt.imshow(area, origin='lower', interpolation='nearest')
    plt.show()
    exit()

if not AREA_SHELVES:
    exit()

if SAVE:
    f = open(FILE_OUT, 'w')
    f.write('ice_shelf,area_km2,sampled_km2,sampled_%,n_cells\n')

# calculate ice-shelf area
print 'Area in km2'
for k, region in enumerate(shelves):
    if names[k] == 'East_Antarctica':
        area_data1, n_cells_data1 = ap.get_area(data[0], lon, lat, ap.eais1)
        area_mask1, n_cells_mask1 = ap.get_area(mask_data, lon, lat, ap.eais1)
        area_data2, n_cells_data2 = ap.get_area(data[0], lon, lat, ap.eais2)
        area_mask2, n_cells_mask2 = ap.get_area(mask_data, lon, lat, ap.eais2)
        area_data = area_data1 + area_data2
        area_mask = area_mask1 + area_mask2
        n_cells_data = n_cells_data1 + n_cells_data2
    else:
        area_data, n_cells_data = ap.get_area(data[0], lon, lat, region)
        area_mask, n_cells_mask = ap.get_area(mask_data, lon, lat, region)

    percentage = (area_data / area_mask) * 100
    print '%s:  area = %.2f,  sampled = %.2f (%%%.1f),  #cells = %g' \
          % (names[k], area_mask, area_data, percentage, n_cells_data)

    if SAVE:
        f.write('%s,%.2f,%.2f,%.1f,%g\n' \
                % (names[k], area_mask, area_data, percentage, n_cells_data))

    if 0:
        plt.figure()
        plt.imshow(mask_data, extent=(lon.min(), lon.max(), lat.min(), lat.max()), 
                   origin='lower', interpolation='nearest', aspect='auto')
        plt.imshow(data[10], extent=(lon.min(), lon.max(), lat.min(), lat.max()), 
                   origin='lower', interpolation='nearest', aspect='auto')
        plt.grid(True)
        plt.show()
        sys.exit()

f.close()
print 'out ->', FILE_OUT
