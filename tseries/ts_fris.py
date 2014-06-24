#!/urs/bin/env python

import os
import sys
import numpy as np
import tables as tb
import netCDF4 as nc
import scipy.ndimage as img
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import interp

sys.path.append('/Users/fpaolo/code/misc')
import util
import viz

# parameters
#---------------------------------------------------------------------

MAX_DH = 3  # for difference (dh)
MIN_NPTS = 6
FROM_YEAR, TO_YEAR = 1991, 2013 

#---------------------------------------------------------------------

fname = sys.argv[1]

#---------------------------------------------------------------------

def plot_ts(ax, x, y):
    x_pol, y_pol = util.poly_fit(x, y, order=3)
    x_lin, y_lin = linear_fit(x, y)
    m, c = linear_fit(x, y, return_coef=True)
    x = x.round(2)
    n = y_lin[0]    # reference to zero
    ax.plot(x, y-n, linewidth=2.5)
    #ax.plot(x_pol, y_pol-n)
    ax.plot(x_lin, y_lin-n, 'r')
    viz.add_inner_title(ax, 'trend = %.1f cm/yr' % (m*100), loc=3)
    return ax

def create_subplots(fig, m=6, n=1):
    iplot = m*100 + n*10 + i
    ax = fig.add_subplot(iplot, axisbg='#FFF8DC')
    ax.xaxis.set_ticks(range(1992, 2014, 2))
    ax.yaxis.set_ticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    plt.xlim(1991.5, 2012.5)
    plt.ylim(-.4, .4)
    plt.ylabel('dh (m)')
    return ax

#---------------------------------------------------------------------

d = get_dh_data(fname)
dh = d['dh']
year = d['year']
lon = d['lon']
lat = d['lat']
x_edges = d['x_edges']
y_edges = d['y_edges']

# boxes
#---------------------------------------------------------------------

k = {}    # grid indices
'''
k[1] = (23,35,9,20)    # Ronne FZ west
k[2] = (23,35,1,9)     # Ronne GL west
k[3] = (35,51,13,23)   # Ronne FZ east
k[4] = (35,51,1,9)     # Ronne GL east
k[5] = (52,70,9,12)     # Filchner FZ
k[6] = (52,70,3,9)      # Filchner GL
'''

k[1] = (24,35,9,18)    # Ronne FZ west
k[2] = (24,35,2,9)     # Ronne GL west
k[3] = (35,50,9,23)   # Ronne FZ east
k[4] = (35,50,1,9)     # Ronne GL east
k[5] = (55,69,9,13)     # Filchner FZ
k[6] = (55,69,3,9)      # Filchner GL

fig1 = plt.figure(figsize=(9,18))

ax = create_subplots(fig1, 6, 1)

#---------------------------------------------------------------------

j = 1
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 2
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 3
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 4
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 5
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 6
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

'''
j = 7
ts = get_ts(dh, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)
viz.add_inner_title(ax[j], 'whole ice shelf', loc=2)
'''

#---------------------------------------------------------------------


fig1.autofmt_xdate()
plt.savefig('ts_fris.png', dpi=150, bbox_inches='tight')
#os.system('cp ts_fris.pdf /Users/fpaolo/posters/scar12/figures/')

plt.show()

for h5f in tb.file._open_files.values():
    h5f.close()
