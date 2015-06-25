import tables as tb

'''
To generate the XDMF to read HDF5 file, see:
/Users/fpaolo/code/postproc/post_proc.py

WITH TIME INFORMATION => for a 3D array being z,y,x = time,lat,lon
'''

# edit here ---------------------------------------------------------

file_in = '/Users/fpaolo/data/shelves/h_postproc.h5.byfirst3_.slabs'
file_out = 'h_raw_final.xmf'

time = '/time'
xyz = '/xyz_nodes'
data = '/data/dh_mean_mixed_const_xcal_ib_sl'  # without the *_nn*
name = 'BS/IB/SL-corrected h(t)'

# TODO Set the units in here: m or cm or...

#---------------------------------------------------------------------

path_to_xyz = file_in + ':' + xyz 
path_to_data = file_in + ':' + data + '_%02d'
data_name = data + '_00'

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
    
     <Attribute Name="NAME" AttributeType="Scalar" Center="Cell">
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
inner = inner.replace('NAME', name)

# get info from hdf5
with tb.open_file(file_in, 'r') as ff:
    coords = ff.root.xyz_nodes
    coord_rows, coord_cols = coords.shape  # N x 3
    ny, nx = ff.get_node(data_name).shape
    time_ = ff.get_node(time)[:]

# write info to xdmf
with open(file_out, 'w') as f:
    f.write(head)
    inner = inner.format(ny+1, nx+1, 1, coord_rows, coord_cols, ny, nx, 0)
    for i, t in enumerate(time_):
        f.write(inner % (t, i))
    f.write(tail)

print 'out ->', file_out
