import sys
import numpy as np
import tables as tb

dbfile = sys.argv[1]

class elevation(tb.IsDescription):
    orbit = tb.Int32Col(pos=1)
    utc85 = tb.Float64Col(pos=2)
    lon = tb.Float64Col(pos=3)
    lat = tb.Float64Col(pos=4)
    elev = tb.Float64Col(pos=5)
    agc = tb.Float64Col(pos=6)
    fmode = tb.Int8Col(pos=7)
    fret = tb.Int8Col(pos=8)
    fprob = tb.Int8Col(pos=9)

h5f = tb.openFile(dbfile, 'w')
filt = tb.Filters(complib='blosc', complevel=9)

g1 = h5f.createGroup('/', 'seasat', 'Seasat data')
g1a = h5f.createGroup(g1, 'elevation', 'Seasat elevation measurements')
g1b = h5f.createGroup(g1, 'crossover', 'Seasat crossover data')
g1c = h5f.createGroup(g1, 'timeseries', 'Seasat time series')

g2 = h5f.createGroup('/', 'geosatgm', 'Geosat/GM data')
g2a = h5f.createGroup(g2, 'elevation', 'Geosat/GM elevation measurements')
g2b = h5f.createGroup(g2, 'crossover', 'Geosat/GM crossover data')
g2c = h5f.createGroup(g2, 'timeseries', 'Geosat/GM time series')

g3 = h5f.createGroup('/', 'geosaterm', 'Geosat/ERM data')
g3a = h5f.createGroup(g3, 'elevation', 'Geosat/ERM elevation measurements')
g3b = h5f.createGroup(g3, 'crossover', 'Geosat/ERM crossover data')
g3c = h5f.createGroup(g3, 'timeseries', 'Geosat/ERM time series')

g4 = h5f.createGroup('/', 'gfo', 'GFO data')
g4a = h5f.createGroup(g4, 'elevation', 'GFO elevation measurements')
g4b = h5f.createGroup(g4, 'crossover', 'GFO crossover data')
g4c = h5f.createGroup(g4, 'timeseries', 'GFO time series')

g5 = h5f.createGroup('/', 'ers1', 'ERS-1 data')
g5a = h5f.createGroup(g5, 'elevation', 'ERS-1 elevation measurements')
g5b = h5f.createGroup(g5, 'crossover', 'ERS-1 crossover data')
g5c = h5f.createGroup(g5, 'timeseries', 'ERS-1 time series')

g6 = h5f.createGroup('/', 'ers2', 'ERS-2 data')
g6a = h5f.createGroup(g6, 'elevation', 'ERS-2 elevation measurements')
g6b = h5f.createGroup(g6, 'crossover', 'ERS-2 crossover data')
g6c = h5f.createGroup(g6, 'timeseries', 'ERS-2 time series')

g7 = h5f.createGroup('/', 'envisat', 'Envisat data')
g7a = h5f.createGroup(g7, 'elevation', 'Envisat elevation measurements')
g7b = h5f.createGroup(g7, 'crossover', 'Envisat crossover data')
g7c = h5f.createGroup(g7, 'timeseries', 'Envisat time series')


h5f.close()
