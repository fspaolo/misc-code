import sys
import numpy as np
import tables as tb

#import psyco
#psyco.bind(readFile)

fname = 'pig_monthly_dh_v3.txt'

def h5_to_dict(fname):
    """Loads an HDF5 file and returns a dict with the HDF5 data in it."""
    f = tb.openFile(fname)
    outdict = {}
    for leaf in f.walkNodes('/', classname='Leaf'):
        outdict[leaf.name] = leaf.read()
    f.close()
    return outdict

def dict2h5(fname, mydict):
    """ Saves all the arrays in a dictionary to an HDF5 file.
    """
    outfile = tb.openFile(fname, mode = "w")
    for key, value in mydict.iteritems():
        if isinstance(value, ndarray):
           outfile.createArray('/', str(key), value)
    outfile.close()

def dict_to_rec(mydict):
    """Converts from a dictionary to a recarray."""
    cols = [col for col in mydict.itervalues()]
    rectype = [(name, col.dtype) for (name, col) in mydict.iteritems()]
    return np.rec.fromarrays(cols, dtype=rectype)

def rec2h5(fname, myrec):
    """Save a recarray to an HDF5 (py)table.
    """
    h5f = tb.openFile(fname, 'w')
    ra = h5f.createTable('/', 'data', myrec)
    h5f.close()


'''
dtype = np.dtype([
    ('year', 'int64'), 
    ('month', 'int64'),
    ('lon', 'float64'),
    ('lat', 'float64'),
    ('dh', 'float64'),
    ('dagc', 'float64'),
    ('std', 'float64'),
    ('mode', 'float64'),
    ])
'''

class TSeries(tb.IsDescription):
    year = tb.Int64Col(pos=1)
    month = tb.Int64Col(pos=2)
    lon = tb.Float64Col(pos=3)
    lat = tb.Float64Col(pos=4)
    dh = tb.Float64Col(pos=5)
    dagc = tb.Float64Col(pos=6)
    std = tb.Float64Col(pos=7)
    mode = tb.Int64Col(pos=8)

h5f = tb.openFile('xovers_db.h5', 'w')
g1 = h5f.createGroup('/', 'PIG', 'Pine Island Bay area')
filt = tb.Filters(complib='blosc', complevel=9)
t1 = h5f.createTable(g1, 'ts1', TSeries, 'Time series 1', filters=filt)

d = np.loadtxt(fname, dtype=t1.dtype)

t1.append(d)
'''
t2 = h5f.createTable(g1, 'ts2', data, 'Time series 2')
g2 = h5f.createGroup('/', 'ANTPEN', 'Antarctic Peninsula')
t3 = h5f.createTable(g2, 'ts3', data, 'Time series 3')
t1.append(data)
'''

h5f.flush()
h5f.close()
print d.dtype
