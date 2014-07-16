#!/usr/bin/env python

filein =  'ssg_d_filt.txt'
fileout =  'ssg_d_filt.txt'
mark_type = 'NaN'
mark_col = 2
time_col = 4

#----------------------------------------------

import numpy as N
from sys import exit

print 'loading data ...'
x = N.loadtxt(filein)
rows,cols = x.shape

print 'verifing NaNs ...'
count = 0
for i in xrange(rows):

    if N.isnan(x[i,mark_col]) == True:

        x[i,mark_col] = 999999
        x[i,time_col] = -1
        count += 1

N.savetxt(fileout, x, fmt='%f', delimiter=' ')

print 'total marked: ', count
print 'output ->', fileout
