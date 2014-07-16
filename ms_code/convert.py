#!/usr/bin/env python

import numpy as N
import optparse
from sys import exit

usage = "python %prog <filein> [options]"
parser = optparse.OptionParser(usage=usage)

parser.add_option('-s',
    dest='cols_asec',
    default='-1',
    help='columns to convert, urad -> arcsec: -s1,2',
    )
parser.add_option('-S',
    dest='cols_asec2',
    default='-1',
    help='columns to convert, urad**2 -> arcsec**2: -s1,2',
    )
parser.add_option('-r',
    dest='cols_urad',
    default='-1',
    help='columns to convert, arcsec -> urad: -r1,2',
    )
parser.add_option('-R',
    dest='cols_urad2',
    default='-1',
    help='columns to convert, arcsec**2 -> urad**2: -R1,2',
    )
parser.add_option('-o',
    dest='fileout',
    default='convert.out',
    help='write the output to FILEOUT: -oconvert.out',
    )
options, remainder = parser.parse_args()


if not len(remainder) == 1:
    parser.print_help()
    exit()

filein = remainder[0]
fileout = options.fileout
cols_s = eval(options.cols_asec)
cols_s2 = eval(options.cols_asec2)
cols_r = eval(options.cols_urad)
cols_r2 = eval(options.cols_urad2)

r = 4.8481368169032191    # 1 arcsec = 4.8481368169032191 urad
r2 = 23.504430595412476   # 1 arcsec**2 = 23.504430595412476 urad**2
s = 1.0/r
s2 = 1.0/r2


print 'loading data ...'
data = N.loadtxt(filein)

if not cols_s == -1:
    for col in cols_s:
        print 'col %d: urad -> arcsec' % col
        data[:,col] *= s

if not cols_s2 == -1:
    for col in cols_s2:
        print 'col %d: urad**2 -> arcsec**2' % col
        data[:,col] *= s2

if not cols_r == -1:
    for col in cols_r:
        print 'col %d: arcsec -> urad' % col
        data[:,col] *= r

if not cols_r2 == -1:
    for col in cols_r2:
        print 'col %d: arcsec**2 -> urad**2' % col
        data[:,col] *= r2

print 'saving data ...'
N.savetxt(fileout, data, fmt='%f', delimiter=' ')
print 'output ->', fileout

