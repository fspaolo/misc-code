import sys
import re
import datetime as dt
import numpy as np
import tables as tb

class Elevation(tb.IsDescription):
    time = tb.StringCol(64, pos=1)
    orbit = tb.Int32Col(pos=2)
    utc85 = tb.Float64Col(pos=3)
    lon = tb.Float64Col(pos=4)
    lat = tb.Float64Col(pos=5)
    elev = tb.Float64Col(pos=6)
    agc = tb.Float64Col(pos=7)
    fmode = tb.Int8Col(pos=8)
    fret = tb.Int8Col(pos=9)
    fprob = tb.Int8Col(pos=10)
    fmask = tb.Int8Col(pos=11)
    fbord = tb.Int8Col(pos=12)
    ftrack = tb.Int8Col(pos=13)
    inc = tb.Float64Col(pos=14)

class Crossover(tb.IsDescription):
    sat = tb.StringCol(32, pos=1)
    region = tb.Int16Col(pos=2)
    lon = tb.Float64Col(pos=3)
    lat = tb.Float64Col(pos=4)
    time_1 = tb.StringCol(64, pos=5)
    time_2 = tb.StringCol(64, pos=6)
    year_1 = tb.Int16Col(pos=7)
    year_2 = tb.Int16Col(pos=8)
    month_1 = tb.Int16Col(pos=9)
    month_2 = tb.Int16Col(pos=10)
    season_1 = tb.StringCol(32, pos=11)
    season_2 = tb.StringCol(32, pos=12)
    orbit_1 = tb.Int32Col(pos=13)
    orbit_2 = tb.Int32Col(pos=14)
    utc85_1 = tb.Float64Col(pos=15)
    utc85_2 = tb.Float64Col(pos=16)
    elev_1 = tb.Float64Col(pos=17)
    elev_2 = tb.Float64Col(pos=18)
    agc_1 = tb.Float64Col(pos=19)
    agc_2 = tb.Float64Col(pos=20)
    fmode_1 = tb.Int8Col(pos=21)
    fmode_2 = tb.Int8Col(pos=22)
    fret_1 = tb.Int8Col(pos=23)
    fret_2 = tb.Int8Col(pos=24)
    fprob_1 = tb.Int8Col(pos=25)
    fprob_2 = tb.Int8Col(pos=26)
    fmask_1 = tb.Int8Col(pos=27)
    fmask_2 = tb.Int8Col(pos=28)
    fbord_1 = tb.Int8Col(pos=29)
    fbord_2 = tb.Int8Col(pos=30)
    ftrack_1 = tb.Int8Col(pos=31)
    ftrack_2 = tb.Int8Col(pos=32)
    inc_1 = tb.Float64Col(pos=33)
    inc_2 = tb.Float64Col(pos=34)

class TimeSeries(tb.IsDescription):
    id = tb.Int16Col(pos=1)
    year = tb.Int16Col(pos=2)
    month = tb.Int16Col(pos=3)
    dh = tb.Float64Col(pos=4)
    se_dh = tb.Float64Col(pos=5)
    dg = tb.Float64Col(pos=6)
    se_dg = tb.Float64Col(pos=7)
    n_ad = tb.Int16Col(pos=8)
    n_da = tb.Int16Col(pos=9)


def utc85_to_datetime(utc85):
    """Converts frac seconds from 1985-1-1 00:00:00 to datetime.

    Note: utc 1985 (or "ESA time") is local time.
    """
    #utc85[np.isnan(utc85)] = 0.0
    datetime85 = dt.datetime(1985, 1, 1)
    return [(datetime85 + dt.timedelta(seconds=fsecs)) for fsecs in utc85]


def season(month):
    """Given a month returns the respective season (SH).
    """
    season1 = [12, 1, 2]     # Dec--Feb -> summer SH
    season2 = [3, 4, 5]      # Mar--May -> fall   SH 
    season3 = [6, 7, 8]      # Jun--Aug -> winter SH
    season4 = [9, 10, 11]    # Sep--Nov -> spring SH

    if month in season1:
        return 'summer'
    elif month in season2:
        return 'fall'
    elif month in season3:
        return 'winter'
    elif month in season4:
        return 'spring'
    else:
        print 'not a valid month from 1 to 12!'
        return None


def fill_table_elevation(table, files):
    print 'filling table ...'
    nan = 0
    for f in files: 
        fin = tb.openFile(f, 'r')
        data = fin.root.data
        datetime = utc85_to_datetime(data[:,1])
        for time, row in zip(datetime, data):
            if np.isnan(row).any():
                nans += 1
                continue
            table.row['time'] = time 
            table.row['orbit'] = row[0]
            table.row['utc85'] = row[1]
            table.row['lon'] = row[3]    # lon first
            table.row['lat'] = row[2]
            table.row['elev'] = row[4]
            table.row['agc'] = row[5]
            table.row['fmode'] = row[6]
            table.row['fret'] = row[7]
            table.row['fprob'] = row[8]
            table.row['fmask'] = row[9]
            table.row['fbord'] = row[10]
            table.row['ftrack'] = row[11]
            table.row['inc'] = row[12]
            table.row.append()
        table.flush()
        fin.close()


def fill_table_crossover(table, data, sat, region):
    datetime1 = utc85_to_datetime(data[:,4])
    datetime2 = utc85_to_datetime(data[:,5])
    nans = 0
    for time1, time2, row in zip(datetime1, datetime2, data):
        if np.isnan(row).any():
            nans += 1
            continue
        table.row['sat'] = sat
        table.row['region'] = region
        table.row['lon'] = row[0]
        table.row['lat'] =  row[1]
        table.row['time_1'] = time1
        table.row['time_2'] = time2
        table.row['year_1'] = time1.year
        table.row['year_2'] = time2.year
        table.row['month_1'] = time1.month
        table.row['month_2'] = time2.month
        table.row['season_1'] = season(time1.month)
        table.row['season_2'] = season(time2.month)
        table.row['orbit_1'] = row[2]
        table.row['orbit_2'] = row[3]
        table.row['utc85_1'] = row[4]
        table.row['utc85_2'] = row[5]
        table.row['elev_1'] = row[6]
        table.row['elev_2'] = row[7]
        table.row['agc_1'] = row[8]
        table.row['agc_2'] = row[9]
        table.row['fmode_1'] = row[10]
        table.row['fmode_2'] = row[11]
        table.row['fret_1'] = row[12]
        table.row['fret_2'] = row[13]
        table.row['fprob_1'] = row[14]
        table.row['fprob_2'] = row[15]
        table.row['fmask_1'] = row[16]
        table.row['fmask_2'] = row[17]
        table.row['fbord_1'] = row[18]
        table.row['fbord_2'] = row[19]
        table.row['ftrack_1'] = row[20]
        table.row['ftrack_2'] = row[21]
        table.row['inc_1'] =  row[22]
        table.row['inc_2'] = row[23]
        table.row.append()
    table.flush()
    return nans


def main():

    filt = tb.Filters(complib='blosc', complevel=9)
    db = tb.openFile('db.h5', 'a')

    #----------------------------------------------------------------
    # create group
    #----------------------------------------------------------------
    #group = db.createGroup('/', 'geosaterm')
    #g1 = db.createGroup(g0, 'elevation')
    #g2 = db.createGroup(g0, 'crossover')
    #g3 = db.createGroup(g0, 'tseries')

    #----------------------------------------------------------------
    # create table
    #----------------------------------------------------------------
    tablename = 'crossover'
    title = 'Crossovers'
    #t = db.createTable(g1, tablename, Elevation, title, filt)
    #table = db.createTable(group, tablename, Crossover, title, filt)
    #t = db.createTable(g3, tablename, TimeSeries, title, filt)

    #----------------------------------------------------------------
    # fill table
    #----------------------------------------------------------------
    files = sys.argv[1:]
    print 'input files:', len(files)
    print 'filling table ...'
    for fname in files: 
        #sat = fname.split('_')[0]
        sat = 'geosaterm'
        region = int(re.search('reg\d\d', fname).group()[-2:]) + 1
        f = tb.openFile(fname, 'r')
        data = f.root.data
        table = db.getNode('/', '%s/crossover' % sat)
        #fill_table_elevation(table, data[1:,:], sat, region)
        nans = fill_table_crossover(table, data[1:,:], sat, region)
        f.close()
    print "NaN's discarded:", nans

    db.close()

if __name__ == '__main__':
    main()
