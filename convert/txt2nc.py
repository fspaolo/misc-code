import os
import sys
import numpy as np
from glob import glob
import netCDF4

def loadnc(fname):
    ncfile = netCDF4.Dataset(fname)
    data = ncfile.variables['data'][:]
    ncfile.close()
    return data

def savenc(fname, data, fmt='NETCDF4'):
    nrow, ncol = np.shape(data)
    ncfile = netCDF4.Dataset(fname, 'w', format=fmt)
    ncfile.createDimension('nrow', nrow)
    ncfile.createDimension('ncol', ncol)
    d = ncfile.createVariable('data', 'f8', ('nrow', 'ncol'))
    d[:] = data
    ncfile.close()

files = sys.argv[1:]
for f in files:
    #data = np.loadtxt(f)
    data = loadnc(f)
    savenc(os.path.splitext(f)[0]+'.nc', data) 

print 'done!' 
