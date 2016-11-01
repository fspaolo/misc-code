import math as M

lon_1 = 0.0         #  0/360
lat_1 = 0.0         # -90/90
lon_2 = 45.0
lat_2 = 45.0

R  = 6371.2  # mean Earth's radius (km)


def deg2rad(lon_1, lat_1, lon_2, lat_2):
    """
    To convert degrees in radians -> degree * pi/180.
    """
    # Converting lon into a range -180/+180.
    if lon_1 > 180:
        lon1 = M.radians(lon_1 - 360) 
    else:
        lon1 = M.radians(lon_1)
    if lon_2 > 180:
        lon2 = M.radians(lon_2 - 360) 
    else:
        lon2 = M.radians(lon_2)
    lat1 = M.radians(lat_1)    
    lat2 = M.radians(lat_2)

    return lon1, lat1, lon2, lat2


def dist_azmth(lon1, lat1, lon2, lat2):
    """
    Azimuth and distance using planar aproximation
    (planar trigonometry).
    """
    latm = (lat1 + lat2) / 2.0
    dx = R * M.cos(latm) * (lon2 - lon1)
    dy = R * (lat2 - lat1)

    # Distance.
    dist = M.hypot(dx, dy)

    # Azimuth (in radians).
    azmth = M.atan2(dx, dy)

    #if dy != 0:
    #    if (dx > 0 and dy > 0):
    #        azmth = M.atan(dx/dy)
    #    elif (dx < 0 and dy < 0) or (dx > 0 and dy < 0):
    #        azmth = M.atan(dx/dy) + M.pi
    #    elif (dx < 0 and dy > 0):
    #        azmth = M.atan(dx/dy) + 2 * M.pi
    #    else:
    #        azmth = None
    #else:
    #    azmth = None

    return dist, azmth


if __name__ == '__main__':
    lon1, lat1, lon2, lat2 = deg2rad(lon_1, lat_1, lon_2, lat_2)
    dist, azmth = dist_azmth(lon1, lat1, lon2, lat2)
    print 'P:(%s, %s)  Q:(%s, %s)' %(M.degrees(lon1), M.degrees(lat1), 
                                     M.degrees(lon2), M.degrees(lat2))
    print 'Distance: ', dist,  'km   or ', dist/111.,        'deg'
    print 'Azimuth:  ', azmth, 'rad  or ', M.degrees(azmth), 'deg'
