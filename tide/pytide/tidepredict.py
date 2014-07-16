import numpy as np


def astrol(t):
    """
    Given t in decimal MJD, returns S, H, P, N.

    Vectorized; each output has the dimensions of the input.

    *======================================================================
          SUBROUTINE ASTROL( time, SHPN )
    *
    *  Computes the basic astronomical mean longitudes  s, h, p, N.
    *  Note N is not N', i.e. N is decreasing with time.
    *  These formulae are for the period 1990 - 2010, and were derived
    *  by David Cartwright (personal comm., Nov. 1990).
    *  time is UTC in decimal MJD.
    *  All longitudes returned in degrees.
    *  R. D. Ray    Dec. 1990
    *
    """
    t = np.asarray(t)

    T = t - 51544.4993
    # mean longitude of moon
    S = 218.3164 + 13.17639648 * T

    # mean longitude of sun
    H = 280.4661 +  0.98564736 * T

    # mean longitude of lunar perigee
    P =  83.3535 +  0.11140353 * T

    # mean longitude of ascending lunar node
    N = 125.0445 -  0.05295377 * T

    S, H, P, N = [arg % 360 for arg in (S, H, P, N)]

    return S, H, P, N

def date_from_jd(J):
    """
    Given Julian Day or Julian Day Number, return year, month, day.

    If J is an integer (i.e., Julian Day Number), the corresponding
    integer day will be
    returned; otherwise, day will be floating point.  Note that
    the JD starts at noon

    From http://www.astro.uu.nl/~strous/AA/en/reken/juliaansedag.html
    """
    scalar = not np.iterable(J)
    J = np.atleast_1d(J)
    if J.dtype.kind in 'iu':
        j = J - 1721120
        dfrac = 0
    else:
        j = (J - 1721119.5).astype(int)
        dfrac = J - j - 1721119.5
    # Now we can do everything with integer arithmetic.
    c7 = j // 146097
    x6 = j % 146097
    c6 = x6 // 36524
    c6 -= (c6 // 4)
    x5 = x6 - 36524 * c6
    c5 = x5 // 1461
    x4 = x5 % 1461
    c4 = x4 // 365
    c4 -= (c4 // 4)
    x3 = x4 - 365 * c4
    c3 = x3 // 153
    x2 = x3 % 153
    c2 = x2 // 61
    x1 = x2 % 61
    c1 = x1 // 31
    x0 = x1 % 31
    y = 400*c7 + 100 * c6 + 4*c5 + c4
    m = 5*c3 + 2*c2 + c1 + 3
    d = x0 + 1 + dfrac
    cond = (m > 12)
    if cond.any():
        m[cond] -= 12
        y[cond] += 1
    if scalar:
        y, m, d = y[0], m[0], d[0]
    return y, m, d



def jd_from_date(y, m, d):
    """
    Given year, month, day, return the Julian Day.

    If d is an integer, the JD number for that day is returned
    as an integer; if d is floating point, it is assumed to be
    day + fraction, and the corresponding floating point JDN is
    returned.  Note that the fractional part of this is zero at
    noon, so

    int(jd_from_date(2000, 10, 10.5)) == jd_from_date(2000, 10, 10)

    Vectorized: y, m, d can be broadcast-compatible sequences.

    Note: for scalars, a python scalar version would be much faster;
    but this optimization would likely not matter in practice.

    From http://www.astro.uu.nl/~strous/AA/en/reken/juliaansedag.html
    """
    if np.iterable(y) or np.iterable(m) or np.iterable(d):
        scalar = False
    else:
        scalar = True
    y, m, d = np.atleast_1d(y, m, d)
    y = y.astype(int)
    m = m.astype(int)
    if d.dtype.kind in 'iu':
        day_offset = 1721119
    else:
        day_offset = 1721118.5

    cond = m < 3
    if cond.any():
        m += 12
        y -= 1
    mm3 = m - 3
    c7 = y // 400
    x6 = y % 400
    c6 = x6 // 100
    x5 = x6 % 100
    c5 = x5 // 4
    c4 = x5 % 4
    c3 = mm3 // 5
    x2 = mm3 % 5
    c2 = x2 // 2
    c1 = x2 % 2
    jd = (146097 * c7 + 36524 * c6 + 1461 * c5 + 365 * c4 + 153 * c3
            + 61 * c2 + 31 * c1 + d + day_offset)

    if scalar:
        jd = jd[0]
    return jd

def mjd_from_date(y, m, d):
    """
    Given year, month, day, return Modified Julian Day.

    If day is an integer, the corresponding integer MJD is
    returned; if day is floating point, MJD is returned as
    floating point with the same fractional part of a day.

    Vectorized: y, m, d can be broadcast-compatible sequences.

    See http://en.wikipedia.org/wiki/Julian_day
    and http://tycho.usno.navy.mil/mjd.html

    """
    jd = jd_from_date(y, m, d)
    if jd.dtype.kind in 'iu':
        offset = 2400001
    else:
        offset = 2400000.5
    return jd - offset

def mjd_from_dday(yearbase, dday):
    return mjd_from_date(yearbase, 1, 1.0) + dday

def date_from_mjd(mjd):
    """
    Given Modified Julian Day, return year, month day.

    See mjd_from_date and date_from_jd.
    """
    scalar = not np.iterable(mjd)
    mjd = np.atleast_1d(mjd)
    if mjd.dtype.kind in 'iu':
        offset = 2400001
    else:
        offset = 2400000.5
    jd = mjd + offset
    y, m, d = date_from_jd(jd)
    if scalar:
        y, m, d = y[0], m[0], d[0]
    return y, m, d

psmap_doc = """
function [x,y]= mapll(lat,lon,SLAT,SLON,HEMI);
%*************************************************************************
%
%    DESCRIPTION:
%
%    This function converts from geodetic latitude and longitude to Polar
%    Stereographic (X,Y) coordinates for the polar regions.  The equations
%    are from Snyder, J. P., 1982,  Map Projections Used by the U.S.
%    Geological Survey, Geological Survey Bulletin 1532, U.S. Government
%    Printing Office.  See JPL Technical Memorandum 3349-85-101 for further
%    details.
%
%    ARGUMENTS:
%
%    Variable     I/O    Description
%
%    lat           I     Geodetic Latitude (degrees, +90 to -90)
%    lon           I     Geodetic Longitude (degrees, 0 to 360)
%    SLAT          I     Standard latitude (typ. 71, or 70)
%    SLON          I
%    HEMI          I     Hemisphere (char*1: 'N' or 'S' (not
%                                    case-sensitive)
%    x             O     Polar Stereographic X Coordinate (km)
%    y             O     Polar Stereographic Y Coordinate (km)
%
%
% FORTRAN CODE HISTORY
%    Written by C. S. Morris - April 29, 1985
%    Revised by C. S. Morris - December 11, 1985
%    Revised by V. J. Troisi - January 1990
%       SGN - provides hemisphere dependency (+/- 1)
%    Revised by Xiaoming Li - October 1996
%    Corrected equation for RHO
%
%  Converted from FORTRAN to Matlab by L. Padman - 25-Oct-2006
%  Updated for SLON                 by L. Padman - 21-Nov-2006
%
    Converted to python by EF 2010/12/02
"""

def polar_stereo(lon, lat, SLON, SLAT):
    CDR=57.29577951
    E2 = 6.694379852e-3           # Eccentricity squared
    E  = np.sqrt(E2)
    pi = 3.141592654
    RE=6378.1370                  # Updated 2/11/08 (see email from
                                  #       Shad O'Neel)

    if abs(SLAT) == 90:
        RHO=2*RE/((1+E)**(1+E) * (1-E)**(1-E))**(E/2)
    else:
        SL  = abs(SLAT)/CDR
        TC  = np.tan(np.pi/4-SL/2)/((1-E*np.sin(SL))/(1+E*np.sin(SL)))**(E/2)
        MC  = np.cos(SL)/np.sqrt(1-E2*(np.sin(SL)**2))
        RHO = RE*MC/TC

    lat = np.abs(lat)/CDR
    T   = np.tan(pi/4-lat/2)/((1-E*np.sin(lat))/(1+E*np.sin(lat)))**(E/2)
    lon =-(lon-SLON)/CDR
    x   =-RHO*T*np.sin(lon)
    y   = RHO*T*np.cos(lon)
    return  x, y

def ll_xy_CATS2008(lon, lat):
    i_out = lat > -40
    if i_out.any():
        lon = np.ma.masked_where(i_out, lon)
        lat = np.ma.masked_where(i_out, lat)
    return polar_stereo(lon, lat, -70, 71)

mapfuncs = {}
mapfuncs[4] = ll_xy_CATS2008



