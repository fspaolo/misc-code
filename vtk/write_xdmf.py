import tables as tb

'''
To generate the XDMF to read HDF5 file, see:
/Users/fpaolo/code/postproc/post_proc.py

WITH TIME INFORMATION!
'''

# edit here ---------------------------------------------------------

file_in = '/Users/fpaolo/data/shelves/h_trendfit.h5'
file_out = 'poly_lasso.xmf'

path_to_xyz = file_in + ':' + '/xyz_nodes'
#path_to_data = file_in + ':' + '/data/dh_mean_mixed_const_xcal_%02d'
#var_name = '/data/dh_mean_mixed_const_xcal_00'
path_to_data = file_in + ':' + '/poly_lasso_slabs/poly_lasso_%02d'
var_name = '/poly_lasso_slabs/poly_lasso_00'
time_name = '/time'
data_name = 'h(t)'

#---------------------------------------------------------------------

# XDMF template (do not edit)

head = """\
<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.1">
<Domain>
<Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">

"""

inner = """\
   <!-- time step -->
   <Grid Name="Mesh" GridType="Uniform">
     <Time Value="%f" />
     <Topology TopologyType="3DSMesh" NumberOfElements="{0} {1} {2}"/>
     <Geometry GeometryType="XYZ">
       <DataItem Name="Coordinates" Dimensions="{3} {4}" NumberType="Float" Precision="4" Format="HDF">
         PATH_TO_XYZ
       </DataItem>
     </Geometry>
    
     <Attribute Name="DATA_NAME" AttributeType="Scalar" Center="Cell">
       <DataItem Dimensions="{5} {6} {7}" NumberType="Float" Precision="4" Format="HDF">
         PATH_TO_DATA
       </DataItem>
     </Attribute>
   </Grid>

"""

tail = """\
</Grid>
</Domain>
</Xdmf>
"""

inner = inner.replace('PATH_TO_XYZ', path_to_xyz)
inner = inner.replace('PATH_TO_DATA', path_to_data)
inner = inner.replace('DATA_NAME', data_name)

# get info from hdf5
with tb.open_file(file_in, 'r') as ff:
    coords = ff.root.xyz_nodes
    coord_rows, coord_cols = coords.shape  # N x 3
    ny, nx = ff.get_node(var_name).shape
    time = ff.get_node(time_name)[:]

# write info to xdmf
with open(file_out, 'w') as f:
    f.write(head)
    inner = inner.format(ny+1, nx+1, 1, coord_rows, coord_cols, ny, nx, 0)
    for i, t in enumerate(time):
        f.write(inner % (t, i))
    f.write(tail)
