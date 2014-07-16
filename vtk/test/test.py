import sys
import numpy as np
import tables as tb
import altimpy as ap
import matplotlib.pyplot as plt
from mayavi import mlab
from tvtk.api import tvtk


def rect_grid():
    data = np.random.random((3, 3, 3))
    r = tvtk.RectilinearGrid()
    r.point_data.scalars = data.ravel()
    r.point_data.scalars.name = 'scalars'
    r.dimensions = data.shape
    r.x_coordinates = np.array((0, 0.7, 1.4))
    r.y_coordinates = np.array((0, 1, 3))
    r.z_coordinates = np.array((0, .5, 2))
    print data.ravel().shape
    print data.shape
    print np.shape(r.x_coordinates)
    return r


f = tb.openFile('/Users/fpaolo/data/shelves/all_elev.h5')
f2 = tb.openFile('/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5')
#elev = f.root.elev[:]
elev = f2.root.dh_mean_all_interp_corr_short_tint[:]
lon = f2.root.x_edges[:]
lat = f2.root.y_edges[:]

lon, lat = np.meshgrid(lon, lat)
lon = lon.ravel()
lat = lat.ravel()

#plt.plot(lon, lat, '.')
plt.imshow(elev[10,...], origin='lower')
plt.show()

x, y, z = ap.sph2xyz(lon, lat)

'''
points = mlab.points3d(x, y, z,
                     scale_mode='none',
                     scale_factor=0.01,
                     color=(0, 0, 1))

mlab.show()
'''

'''
points = np.column_stack((x,y,z)) 
#triangles = array([[0,1,3], [0,3,2], [1,2,3], [0,2,1]])
#temperature = array([10., 20., 30., 40.])

# The TVTK dataset.
mesh = tvtk.PolyData(points=points)#, polys=triangles)
mesh.point_data.scalars = elev
mesh.point_data.scalars.name = 'Elevation'

# Uncomment the next two lines to save the dataset to a VTK XML file.
w = tvtk.XMLPolyDataWriter(input=mesh, file_name='polydata.vtp')
w.write()
'''

#r = rect_grid()

data = np.atleast_3d(elev[10,...])
r = tvtk.StructuredGrid()
r.dimensions = (data.shape[0]+1, data.shape[1]+1, data.shape[2])
r.points = np.column_stack((x,y,z)) 
r.cell_data.scalars = data.ravel()
r.cell_data.scalars.name = 'scalars'

w2 = tvtk.XMLStructuredGridWriter(input=r, file_name='structgrid.vts')
w2.write()

f2.close()
f.close()
