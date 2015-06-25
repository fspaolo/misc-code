import tables as tb

'''
To generate the XDMF to read HDF5 file, see:
/Users/fpaolo/code/postproc/postproc.py

WITHOUT TIME INFORMATION => FOR COORDS GRID ORIENTED IN SPACE (W/XYZ)!
'''

# edit here ---------------------------------------------------------

file_in = '/Users/fpaolo/data/shelves/h_postproc.h5'
file_out = 'coords_raw.xmf'

path_to_xyz = file_in + ':' + '/xyz_nodes'
path_to_lon = file_in + ':' + '/xx'
path_to_lat = file_in + ':' + '/yy'

#---------------------------------------------------------------------

# XDMF template (do not edit)

var_lon = path_to_lon.split('/')[-1]
var_lat = path_to_lat.split('/')[-1]

template = """\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">
<Xdmf Version="2.0">
  <Domain>
  <Grid Name="Mesh" GridType="Uniform">
    <Topology TopologyType="3DSMesh" NumberOfElements="{0} {1} {2}"/>

    <Geometry GeometryType="XYZ">
      <DataItem Name="Coordinates" Dimensions="{3} {4}" NumberType="Float" Precision="4" Format="HDF">
        PATH_TO_XYZ
      </DataItem>
    </Geometry>

    <Attribute Name="Longitude" AttributeType="Scalar" Center="Cell">
      <DataItem Dimensions="{5} {6} {7}" NumberType="Float" Precision="4" Format="HDF">
        PATH_TO_LON
      </DataItem>
    </Attribute>

    <Attribute Name="latitude" AttributeType="Scalar" Center="Cell">
      <DataItem Dimensions="{8} {9} {10}" NumberType="Float" Precision="4" Format="HDF">
        PATH_TO_LAT
      </DataItem>
    </Attribute>

  </Grid>
  </Domain>
</Xdmf>
"""

template = template.replace('PATH_TO_XYZ', path_to_xyz)
template = template.replace('PATH_TO_LON', path_to_lon)
template = template.replace('PATH_TO_LAT', path_to_lat)

# get info from hdf5
with tb.openFile(file_in, 'r') as ff:
    coords = ff.root.xyz_nodes
    coord_rows, coord_cols = coords.shape  # N x 3
    ny, nx = getattr(ff.root, var_lon).shape

# write info to xdmf
with open(file_out, 'w') as f:
    template = template.format(ny+1, nx+1, 1, coord_rows, coord_cols, ny, nx, 0, ny, nx, 0)
    f.write(template)
