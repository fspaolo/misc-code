
import os
import sys
import numpy as np
import tables as tb
import matplotlib.pyplot as plt

# warnning: y, x !!!
def bin_by_mean(lon, lat, z, bins=10, range=None):
    bins = bins[::-1] 
    range = range[::-1]
    w_sum, _ = np.histogramdd((lat, lon), weights=z, bins=bins, range=range)
    n_pts, edges = np.histogramdd((lat, lon), bins=bins, range=range) 
    n_pts[n_pts==0] = np.nan
    return (w_sum/n_pts), n_pts, edges[1], edges[0]

files = sys.argv[1:]

for fname in files:
    f = tb.openFile(fname)
    data = f.root.data
    lon, lat = data[:,0], data[:,1]
    h1, h2 = data[:,6], data[:,7]
    g1, g2 = data[:,8], data[:,9]
    f1, f2 = data[:,20], data[:,21]

    #------------------------------------------------------------

    x_range = (-100, -95)
    y_range = (-72.5, -71)
    nx = 10 
    ny = 5

    z_mean, n_pts, x_edges, y_edges = \
            bin_by_mean(lon, lat, h1, (nx,ny), (x_range,y_range))

    #------------------------------------------------------------

    '''
    filters = tb.Filters(complib='blosc', complevel=9)
    atom = tb.Atom.from_dtype(data.dtype)
    
    shape = (data.shape[0], data.shape[1] + 2)
    fout = tb.openFile(os.path.splitext(fname)[0] + '_bin.h5', 'w')
    dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                             filters=filters)
    dout[:,:-2] = data[:] 
    dout[:,-2:] = flags[:] 
    fout.close()
    '''


    extent = [x_range[0], x_range[-1], y_range[0], y_range[-1]]
    plt.figure()
    plt.imshow(z_mean, extent=extent, origin='lower', interpolation='nearest')
    #plt.imshow(ppbin, origin='lower')
    plt.colorbar()
    #plt.figure()
    #plt.plot(x_bins, y_bins, 'o')
    plt.show()

    f.close()
