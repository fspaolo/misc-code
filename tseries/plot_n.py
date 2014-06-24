import sys
import numpy as np
import tables as tb
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as img
from mpl_toolkits.basemap import interp

sys.path.append('/Users/fpaolo/code/misc')
import util
import viz

# input
#---------------------------------------------------------------------

fname = sys.argv[1]

# HDF5
f = tb.openFile(fname)
nrows = f.root.table.nrows
n_ad = f.root.n_ad[:]
n_da = f.root.n_da[:]
lon = f.root.lon[:]
lat = f.root.lat[:]
n_ad[np.isnan(n_ad)] = 0
n_da[np.isnan(n_da)] = 0
dh = n_ad + n_da

dh = np.sum(dh, axis=0)
dh[dh==0] = np.nan

# parameters
ABS_VAL = 0.12  # for trend
#ABS_VAL = 1.5  # for difference
GRID_CELL = (1, 5)
#width = 0.9  # gaussian width
width = 1.5  # gaussian width

BBOX_REG = (225, -50, 45, -50)  # whole Antarctica
#BBOX_REG = (-82, -76.2, -26, -79.5)  # FRIS
#BBOX_REG = (-156, -76, 154, -81.2)   # ROSS
#BBOX_REG = (68, -74.5, 70, -67.5)   # AMERY

MFILE = '/Users/fpaolo/data/masks/scripps_antarctica_masks/scripps_antarctica_mask1km_v1.tif'

#CMAP = cm.jet_r # cm.gist_rainbow #cm.spectral_r # cm.RdBu_r # cm.jet_r
CMAP = viz.colormap('rgb')
LABEL = 'm/yr'

title = 'FRIS 1992-2012\n(ERS1, ERS2, ENVI)'
legend = 'Elevation change rate m/yr'
ticklabels = '0', '4', '8 m/yr'
ticks = -ABS_VAL, 0, ABS_VAL 
colormap = [
    (0, 0.001, 2, 3), # value
    (1, 1, 1, 1), # red
    (0, 0, 1, 1), # green
    (0, 0, 0, 1), # blue
    (0, 1, 1, 1), # alpha
]
colorexp = 1.0
colorlim = ticks[0], ticks[-1]
inches = 6.4, 3.6
extent = (-121.1, -113.8), (32.1, 35.5)
ylim = 150.0
xlim = ylim * inches[0] / inches[1]
axis = -xlim, xlim, -ylim, ylim

print 'lon/lat:', lon[GRID_CELL[1]], lat[GRID_CELL[0]]
#---------------------------------------------------------------------

def plot_map(lon, lat, grid, bbox, mfile, mres=1, **kw):
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = viz.make_proj_stere(bbox)
    m = viz.plot_grid_proj(m, lon, lat, grid, contourf=False, shift=True, **kw)
    #plt.colorbar(orientation='vertical', shrink=.7)
    #x, y, mask = viz.get_gtif_subreg(m, mfile, mres)
    #mask = np.ma.masked_values(mask, 4)
    #m.imshow(mask, zorder=0, cmap=plt.cm.gray_r)
    #m = plot_boundaries(m, (lon.min(),lon.max(),lat.min()-10,lat.max()+3))
    #p_ = m.drawparallels(np.arange(-90.,-70.,3), labels=[0,0,0,1], color='0.6')
    #m_ = m.drawmeridians(np.arange(-80.,-20.,5), labels=[1,0,0,0], color='0.6')
    #viz.colorbar(fig1, CMAP, (-ABS_VAL,ABS_VAL))
    return fig1 


def regrid(lon, lat, dhdt, factor=10):
    m, n = len(lon), len(lat)
    lon2 = np.linspace(lon[0], lon[-1], m*factor)
    lat2 = np.linspace(lat[0], lat[-1], n*factor)
    xx, yy = np.meshgrid(lon2, lat2)
    dhdt2 = interp(dhdt, lon, lat, xx, yy, order=1)                          # good!!!
    return lon2, lat2, dhdt2


#lon, lat, dh = regrid(lon, lat, dh[10,...], factor=2)

plot_map(lon, lat, dh, BBOX_REG, MFILE, mres=1, cmap=cm.jet)
#plt.title('Crossover density per bin (1995/10-2000/10)')
plt.savefig('grid.png',  dpi=150)#, bbox_inches='tight', pad_inches=-0.5)
plt.show()

for fid in tb.file._open_files.values():
    fid.close()
