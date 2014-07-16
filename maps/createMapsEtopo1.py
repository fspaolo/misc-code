import os, sys, datetime, string
import numpy as np
from netCDF4 import Dataset
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid, NetCDFFile
from pylab import *
import laplaceFilter
#import mpl_util

__author__   = 'Trond Kristiansen'
__email__    = 'trond.kristiansen@imr.no'
__created__  = datetime.datetime(2008, 8, 15)
__modified__ = datetime.datetime(2009, 7, 21)
__version__  = "1.0"
__status__   = "Development"

etopo1name='/Users/fpaolo/data/topo/ETOPO1_Ice_g_gmt4.grd'

def findSubsetIndices(min_lat,max_lat,min_lon,max_lon,lats,lons):
    
    """Array to store the results returned from the function"""
    res=np.zeros((4),dtype=np.float64)
    minLon=min_lon; maxLon=max_lon
    
    distances1 = []; distances2 = []
    indices=[]; index=1
    
    for point in lats:
        s1 = max_lat-point # (vector subtract)
        s2 = min_lat-point # (vector subtract)
        distances1.append((np.dot(s1, s1), point, index))
        distances2.append((np.dot(s2, s2), point, index-1))
        index=index+1
        
    distances1.sort()
    distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])
    
    distances1 = []; distances2 = []; index=1
   
    for point in lons:
        s1 = maxLon-point # (vector subtract)
        s2 = minLon-point # (vector subtract)
        distances1.append((np.dot(s1, s1), point, index))
        distances2.append((np.dot(s2, s2), point, index-1))
        index=index+1
        
    distances1.sort()
    distances2.sort()
    indices.append(distances1[0])
    indices.append(distances2[0])
    
    """ Save final product: max_lat_indices,min_lat_indices,max_lon_indices,min_lon_indices"""
    minJ=indices[1][2]
    maxJ=indices[0][2]
    minI=indices[3][2]
    maxI=indices[2][2]
    
    res[0]=minI; res[1]=maxI; res[2]=minJ; res[3]=maxJ;
    return res

def makeMap(lonStart,lonEnd,latStart,latEnd,name,stLon,stLat):
    plt.figure(figsize=(8,8))
   
    """Get the etopo2 data"""
    etopo1 = Dataset(etopo1name,'r')
    
    lons = etopo1.variables["lon"][:]
    lats = etopo1.variables["lat"][:]
    
    res = findSubsetIndices(latStart-5,latEnd+5,lonStart-40,lonEnd+10,lats,lons)
    
    lon,lat=np.meshgrid(lons[res[0]:res[1]],lats[res[2]:res[3]])    
    print "Extracted data for area %s : (%s,%s) to (%s,%s)"%(name,lon.min(),lat.min(),lon.max(),lat.max())
    bathy = etopo1.variables["z"][int(res[2]):int(res[3]),int(res[0]):int(res[1])]
    bathySmoothed = laplaceFilter.laplace_filter(bathy,M=None)
  
    levels=[-6000,-5000,-3000, -2000, -1500, -1000,-500, -400, -300, -250, -200, -150, -100, -75, -65, -50, -35, -25, -15, -10, -5, 0]
        
    if lonStart< 0 and lonEnd < 0:
        lon_0= - (abs(lonEnd)+abs(lonStart))/2.0
    else:
        lon_0=(abs(lonEnd)+abs(lonStart))/2.0
        
    print 'Center longitude ',lon_0
    
    map = Basemap(llcrnrlat=latStart,urcrnrlat=latEnd,\
            llcrnrlon=lonStart,urcrnrlon=lonEnd,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=latStart,lon_0=lon_0)
    
    x, y = map(lon,lat) 
    map.drawcoastlines()
    map.drawcountries()
    map.fillcontinents(color='grey')
    map.drawmeridians(np.arange(lons.min(),lons.max(),10),labels=[0,0,0,1])
    map.drawparallels(np.arange(lats.min(),lats.max(),4),labels=[1,0,0,0])
    #map.bluemarble()

    CS1 = map.contourf(x,y,bathySmoothed,levels,
                       #cmap=mpl_util.LevelColormap(levels,cmap=cm.Blues_r),
                       cmap=cm.Blues_r,
                       extend='upper',
                       alpha=1.0,
                       origin='lower')
    
    CS1.axis='tight'
    """Plot the station as a position dot on the map"""
    xpt,ypt = map(stLon,stLat) 
    map.plot([xpt],[ypt],'ro', markersize=10) 
    plt.text(xpt+100000,ypt+100000,name)
    
    plt.title('Area %s'%(name))
    plotfile='map_'+str(name)+'.pdf'
    plt.savefig(plotfile,dpi=150,orientation='portrait')
    plt.show()
    

def main():
    
    names=['bayOfBiscay']
    lat_start=[40]
    lat_end  =[81]
    
    lon_start=[-10]
    lon_end  =[75]
    
    """List of stations for each area"""
    stationlonlist=[ 2.4301, -22.6001, -47.0801,  13.3801, -67.2001]
    stationlatlist=[54.5601, 63.7010,  60.4201,  67.5001,  41.6423]
    
    for i in range(len(lat_start)):
        if i==0:
            makeMap(lon_start[i],lon_end[i],lat_start[i],lat_end[i],names[i],stationlonlist[i],stationlatlist[i])
        
        
if __name__ == "__main__":

    main()
