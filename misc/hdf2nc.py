import sys
import numpy as np
import tables as tb
import netCDF4 as nc

def int2ymd(iyear):
    f, y = np.modf(iyear/10000.)
    d, m = np.modf(f*100)
    return (int(y), int(m), int(d*100))


def num2year(iyear):
    """Numeric representation of year to decimal year."""
    iyear = np.asarray([int(y) for y in iyear])
    fyear = lambda y, m, d: y + (m - 1)/12. + d/365.25
    ymd = [int2ymd(iy) for iy in iyear]
    return [fyear(y,m,d) for y,m,d in ymd]


if len(sys.argv) < 3:
    raise IOError('need `input` and `output` file names!')

fname_in = sys.argv[1]
fname_out = sys.argv[2]

h5f = tb.openFile(fname_in, 'r')
time = h5f.root.time_all[:]
lon = h5f.root.lon[:]
lat = h5f.root.lat[:]
dh = h5f.root.dh_mean_all[:]
dh_corr = h5f.root.dh_mean_corr_short_t9_all[:]
dg = h5f.root.dg_mean_all[:]
nt, ny, nx = dh.shape

time = num2year(time)
#lon[lon>180] -= 360

ncf = nc.Dataset(fname_out, 'w', format='NETCDF4')
ncf.createDimension('time', nt)
ncf.createDimension('latitude', ny) 
ncf.createDimension('longitude', nx)

dh2 = ncf.createVariable('dh','f8',('time', 'latitude', 'longitude',))
dh_corr2 = ncf.createVariable('dh_corr','f8',('time', 'latitude', 'longitude',))
dg2 = ncf.createVariable('dg','f8',('time', 'latitude', 'longitude',))
time2 = ncf.createVariable('time','f8',('time',))
lon2 = ncf.createVariable('longitude','f8',('longitude',))
lat2 = ncf.createVariable('latitude','f8',('latitude',))


dh[np.abs(dh)>5] = np.nan
dh_corr[np.abs(dh_corr)>10] = np.nan

'''
dh[dh==0] = np.nan
dh_corr[dh_corr==0] = np.nan
dg[dg==0] = np.nan
'''
dh[np.isnan(dh)] = 0
dh_corr[np.isnan(dh_corr)] = 0
dg[np.isnan(dg)] = 0 

dh2[:] = dh[:]
dh_corr2[:] = dh_corr[:]
dg2[:] = dg[:]
time2[:] = time[:]
lon2[:] = lon[:]
lat2[:] = lat[:]

h5f.close()
ncf.close()
