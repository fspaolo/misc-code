import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

fname = '/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5'

with tb.openFile(fname) as f:

    h_firn = f.root.h_firn[:]
    lon = f.root.lon[:]
    lat = f.root.lat[:]

    firn_range = np.nanmax(h_firn, axis=0) - np.nanmin(h_firn, axis=0)
    firn_range[np.abs(firn_range)>1.5] = np.nan

    plt.imshow(firn_range, origin='lower', interpolation='nearest', 
               extent=(lon[0], lon[-1], lat[0], lat[-1]), aspect=4)
    plt.colorbar(orientation='horizontal', shrink=0.6)
    plt.grid(True)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()
