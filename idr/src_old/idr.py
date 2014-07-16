"""
 Definition of IDR structures.

 Fernando Paolo <fpaol@ucsd.edu>
 August 19, 2010
"""

IDRD = [                  # 100-byte IDR Data Record
    ('id', '|S2'),	
    ('retstat1', '>u2'),  # flagwords (bitarrays) are unsigned int !
    ('time', '>i4'), 
    ('lat', '>i4'),  
    ('lon', '>i4'),	  
    ('surf', '>i4'),	  
    ('wdr', '>i4'),	
    ('altim', '>i4'),	  
    ('altstat', '>u4'), 
    ('surfstat', '>u4'),
    ('iono', '>i2'),
    ('wet1', '>i2'),	
    ('dry', '>i2'),
    ('geoid', '>i2'),	
    ('stide', '>i2'),	
    ('otide', '>i2'),	
    ('slope', '>i2'),	
    ('swh', '>i2'),
    ('agc', '>i2'),	
    ('att', '>i2'),	
    ('empty1', '>i2'),
    ('inc1', '>i2'),
    ('empty2', '>i2'),
    ('inc2', '>i2'),
    ('empty3', '>i2'),
    ('inc3', '>i2'),
    ('retramp1', '>i2'),
    ('retramp2', '>i2'),
    ('sigramp1', '>i2'),
    ('sigramp2', '>i2'),
    ('tanslope', '>i2'),
    ('empty4', '>i2'),
    ('wet2', '>i2'),
    ('modstat', '>u2'),	 
    ('locstat', '>u2'),	 
    ('rshstat', '>u2'),	 
    ('wwlstat', '>u2'),	 
    ('opsstat', '>u2'),	 
    ('thres10', '>i2'),
    ('thres20', '>i2'),
    ('thres50', '>i2'),
    ('retstat2', '>u2'),
] 

IDRH = [          # 100-byte IDR Processing Record
    ('id', 'S2'),
    ('revdir', 'S14'),
    ('geodir', 'S14'),
    ('bindir', 'S14'),
    ('version', 'i4'),
    ('datebeg', 'i4'),
    ('timebeg', 'i4'),
    ('dateend', 'i4'),
    ('timeend', 'i4'),
    ('satid', 'i4'),
    ('coverage', 'S8'),
    ('spares', 'S24'),
]

IDRP = [          # 100-byte IDR Processing Record
    ('id', 'S2'),
    ('dateproc', 'S6'),
    ('procprog', 'S18'),
    ('infile1', 'S14'),
    ('infiles', 'S14'),
    ('spares', 'S46'),
]

# time convertion functions -------------------------------------

# http://asimpleweblog.wordpress.com/2010/06/20/julian-date-calculator/

import math


def sexag2deci(xyz, delimiter=None):
    """Decimal value from numbers in sexagesimal system. 
    
    The input value can be either a floating point number or a string
    such as "hh mm ss.ss" or "dd mm ss.ss". Delimiters other than ' '
    can be specified using the keyword ``delimiter``.

    Obs: this function is used by other ones, do not chage it!
    """
    divisors = [1, 60.0, 3600.0]
    xyzlist = str(xyz).split(delimiter)
    sign = -1 if xyzlist[0].find("-") != -1 else 1
    xyzlist = [abs(float(x)) for x in xyzlist]
    decimal_value = 0 
    
    for i,j in zip(xyzlist, divisors): # if xyzlist has <3 values then
                                       # divisors gets clipped.
        decimal_value += i/j
    
    decimal_value = -decimal_value if sign == -1 else decimal_value
    return decimal_value


def deci2sexag(deci, precision=1e-8):
    """Converts decimal number into sexagesimal number parts. 
    
    ``deci`` is the decimal number to be converted. ``precision`` is how
    close the multiple of 60 and 3600, for example minutes and seconds,
    are to 60.0 before they are rounded to the higher quantity, for
    example hours and minutes.
    """
    sign = "+" # simple putting sign back at end gives errors for small
               # deg. This is because -00 is 00 and hence ``format``,
               # that constructs the delimited string will not add '-'
               # sign. So, carry it as a character.
    
    if deci < 0:
        deci = abs(deci)
        sign = "-" 
    
    frac1, num = math.modf(deci)
    num = int(num) # hours/degrees is integer valued but type is float
    frac2, frac1 = math.modf(frac1*60.0)
    frac1 = int(frac1) # minutes is integer valued but type is float
    frac2 *= 60.0 # number of seconds between 0 and 60 
    
    # Keep seconds and minutes in [0 - 60.0000)
    if abs(frac2 - 60.0) < precision:
        frac2 = 0.0
        frac1 += 1
    if abs(frac1 - 60.0) < precision:
        frac1 = 0.0
        num += 1 
    
    return sign, num, frac1, frac2


def date2jd(year, month, day, hour, minute, second):
    """Given year, month, day, hour, minute and second return JD.
      
    ``year``, ``month``, ``day``, ``hour`` and ``minute`` are integers,
    truncates fractional part; ``second`` is a floating point number.
    For BC year: use -(year-1). Example: 1 BC = 0, 1000 BC = -999.
    """
    MJD0 = 2400000.5  # 1858 November 17, 00:00:00 hours 
    
    year, month, day, hour, minute = \
    int(year), int(month), int(day), int(hour), int(minute)
    
    if month <= 2:
        month +=12
        year -= 1 
    
    modf = math.modf
    # Julian calendar on or before 1582 October 4 and Gregorian calendar
    # afterwards.
    if ((10000L * year + 100L * month + day) <= 15821004L):
        b = -2 + int(modf((year + 4716)/4)[1]) - 1179
    else:
        b = int(modf(year/400)[1]) - int(modf(year/100)[1])+ \
        int(modf(year/4)[1]) 
    
    mjdmidnight = 365L*year - 679004L + b + int(30.6001*(month + 1)) + day
    
    fracofday = sexag2deci(" ".join([str(hour),str(minute),str(second)])) / 24.0 
    
    return MJD0 + mjdmidnight + fracofday


def date2mjd(year, month, day, min, hour, sec):
    """Given year, month, day, hour, minute and second return MJD.
    
    This is done using the function date2jd(), see its documentation.
    """
    return date2jd(year, month, day, min, hour, sec) - 2400000.5


def jd2mjd(jd):
    """Given Julian Date return Modified Julian Date.
    
    Julian Day Number is an integer counter of the days beginning at 
    noon on January 1, 4713 BC, which is Julian Day Number 0. The Julian 
    "Date" (as opposed to Julian "Day") is the non-integer extension of 
    the Day Number to include a real fraction of day, allowing a 
    continuous time unit. MJD modifies this Julian Date in two ways. 
    The MJD begins at midnight rather than noon (the .5 in the formula), 
    in keeping with more standard conventions and modern representation 
    of time. Secondly, for simplicity, the first two digits of the Julian 
    Date are removed. This is because, for some three centuries following 
    November 17, 1858, the Julian day lies between 2400000 and 2500000. 
    The MJD drops those first "24" digits: MJD = JD - 2400000.5.
    """
    return jd - 2400000.5 


def utc2mjd(utc85):
    """Given utc85 (ESA time) return Modified Julian Date.
    
    Here utc85 is seconds passed since 1985 January 1 00:00:00 hours
    local time (or ESA time). Do not confuse with UTC (Coordinated 
    Universal Time).
    """
    MJD85 = 46066.    # 1-Jan-1985 00:00:00h in MJD
    DAYSECS = 86400.  # 1 day in seconds 
    mjd = (utc85 / DAYSECS) + MJD85
    return mjd


def mjd2utc(mjd):
    """Given Modified Julian Date return utc85 (ESA time)."""
    MJD85 = 46066.    # 1-Jan-1985 00:00:00h in MJD
    DAYSECS = 86400.  # 1 day in seconds 
    utc85 = (mjd - MJD85) * DAYSECS
    return utc85


def mjd2jd(mjd):
    """Given Modified Julian Date return Julian Date."""
    return mjd + 2400000.5
    

def mjd2date(mjd):
    """Given mjd return calendar date. 
    
    Returns a tuple (year, month, day, hour, minute, second). The last 
    is a floating point number and others are integers. The precision 
    in seconds is about 1e-4. 
    
    To convert jd to mjd use jd - 2400000.5. In this module 2400000.5 
    is stored in MJD0.
    """
    MJD0 = 2400000.5  # 1858 November 17, 00:00:00 hours
    
    modf = math.modf
    a = long(mjd + MJD0 + 0.5)
    # Julian calendar on or before 1582 October 4 and Gregorian 
    # calendar afterwards.
    if a < 2299161:
        b = 0
        c = a + 1524
    else:
        b = long((a - 1867216.25) / 36524.25)
        c = a + b - long(modf(b/4)[1]) + 1525 
    
    d = long((c - 122.1) / 365.25)
    e = 365 * d + long(modf(d/4)[1])
    f = long((c - e) / 30.6001)
    
    day = c - e - int(30.6001 * f)
    month = f - 1 - 12 * int(modf(f/14)[1])
    year = d - 4715 - int(modf((7 + month) / 10)[1])
    fracofday = mjd - math.floor(mjd)
    hours = fracofday * 24.0 
    
    sign, hour, minute, second = deci2sexag(hours)
    return (year, month, day, int(sign+str(hour)), minute, second)


def testdate(year, month, day, hour, min, sec):
    """Test the time conversion functions.

    Given: year, month, day, hour, min, sec, converts to JD, MJD,
    Calendar Date, Hours in Decimal Number, Hours in Sexagesimal Parts,
    ESA time (utc85) and MJD from utc85 (just to check).
    """
    print 'Given values:', year, month, day, hour, min, sec
    jd = date2jd(year, month, day, hour, min, sec)
    print 'Julian Date [date2jd]:', jd
    mjd = date2mjd(year, month, day, hour, min, sec)
    print 'Modified Julian Date [date2mjd]:', mjd
    date = mjd2date(mjd)
    print 'Calendar Date [mjd2date]: %d-%d-%d %d:%d:%.4f hours' % date
    deci = sexag2deci("%d %d %f" % (hour, min, sec))
    print 'Decimal Number [sexag2deci]: %f hours' % deci
    sexag = deci2sexag(deci)
    print 'Sexagesimal Parts [deci2sexag]:', sexag
    utc85 = mjd2utc(mjd)
    print 'ESA time (utc85) [mjd2utc]: %f seconds' % utc85
    mjd = utc2mjd(utc85)
    print 'Modified Julian Date [utc2mjd]:', mjd
