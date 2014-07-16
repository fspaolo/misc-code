def polydata():
    # The numpy array data.
    points = array([[0,-0.5,0], [1.5,0,0], [0,1,0], [0,0,0.5],
                    [-1,-1.5,0.1], [0,-1, 0.5], [-1, -0.5, 0],
                    [1,0.8,0]], 'f')
    scalars = random.random(points.shape)

    # The TVTK dataset.
    mesh = tvtk.PolyData(points=points, polys=triangles)
    mesh.point_data.scalars = scalars
    mesh.point_data.scalars.name = 'scalars'
    return mesh
