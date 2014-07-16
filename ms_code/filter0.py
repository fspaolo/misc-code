#!/usr/bin/env python

dir = '/dados3/fspaolo/DATA/norte/geogm/'

files = [dir+'ssh_a', dir+'ssh_d']

# 1D Gaussian filter with d = 14 km
filter = 'g14k'

# times * sigma
times = 3


import gmtpy
import numpy as N
import pylab as P
import shutil as sh
from subprocess import call

gmt = gmtpy.GMT()

for f in files:
    
    sh.copy(f+'.txt', f+'.tmp')
	    
    iter = 1
    outliers = 0
    repited = 0

    # filtering

    print 'filtering ...'

    gmt.filter1d(
        N='4/3',
        F=filter,
        E=True,
    	V=True,
        in_filename=f+'.tmp',
        out_filename=f+'_filt.txt',
        )
    
    f_init = N.loadtxt(f+'.tmp')
    f_filt = N.loadtxt(f+'_filt.txt')
    
    # outliers

    print 'removing outliers ...'
	 
    rows = len(f_init)
    
    diff = (f_init[:,2] - f_filt[:,2]).copy()
    
    sigma = N.std(diff)
    
    cond = N.ones(rows, 'i')

    for i in xrange(rows):
        if abs(diff[i]) > abs(times * sigma):
            cond[i] = 0
            outliers += 1

    # repited 

    print 'removing repited ...'

    for j in xrange(rows-1):
        if (f_filt[j+1,0] == f_filt[j,0]) and (f_filt[j+1,1] == f_filt[j,1]):
            if not cond[j+1] == 0:   
                cond[j+1] = 0
                repited += 1


    #f_init = N.compress(cond, f_init, 0)
    f_filt = N.compress(cond, f_filt, 0)
    #N.savetxt(f+'.tmp', f_init, fmt='%f', delimiter=' ')
    N.savetxt(f+'_filt.txt', f_filt, fmt='%f', delimiter=' ')
    
    print 'iterations: ', iter
    print 'initial data: ', rows
    print 'outliers: ', outliers
    print 'repited: ', repited
    print 'total removed: ', outliers + repited
        
    fp = open(f+'_filt.info', 'w')
    fp.write('iterations: %d\noutliers: %d\nrepited: %d' % (iter, outliers, repited))
    fp.close()

    print 'output ->',  f+'_filt.txt'
    print 'output ->',  f+'_filt.info'
