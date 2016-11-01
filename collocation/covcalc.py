#!/usr/bin/env python

#    covcalc.py - Program to calculate the covariences between elementes 
#    apart of several distances.
#    
#    It can be calculated
#    1. One file: autocovariance between elements of one file
#    2. Two files: crosscovariance between elementes of two different files
#
#    Author:	Fernando S. Paolo 
#    Date:		jul/2008.
#    Usage:		python covcalc.py -h

import numpy as N
import pylab as P
import optparse
from os import path
from sys import argv, exit
from scipy import weave
from scipy.weave.converters import blitz

# Scan comand line arguments ----------------------------------------------

usage = "python %prog <file1> [file2] [options]"
epilog = "[One file: Autocov | Two files: Crosscov] Units: lon (deg), " + \
         "lat (deg), esph_dist (deg), SSH (m), grav (mGal), " + \
         "geoid (m), cov_xx (x**2), cov_xy (x*y)"
parser = optparse.OptionParser(usage=usage, epilog=epilog)

parser.add_option('-m',
    	          dest='cols1',
                  default='(0,1,2)', 
                  help='columns to be loaded from file1 [lon,lat,x]: -m0,1,2',
                  )
parser.add_option('-n',
    	          dest='cols2',
                  default='(0,1,2)', 
                  help='columns to be loaded from file2 [lon,lat,y]: -n0,1,2',
                  )
parser.add_option('-o', 
    	          dest='fileout',
                  default='covcalc.out', 
                  help='write output to FILEOUT [deg,cov,nprod]: -ocovcalc.out',
                  )
parser.add_option('-k', 
    	          dest='ndist',
                  default=20, 
				  type='int',
                  help='number of dist classes to calculate the covs: -k20',
                  )
parser.add_option('-b', 
    	          dest='begin',
                  default=0.0, 
				  type='float',
                  help='beginning of the interval calculation (deg): -b0.0',
                  )
parser.add_option('-e', 
    	          dest='end',
                  default=2.1, 
				  type='float',
                  help='ending of the interval calculation (deg): -e2.0',
                  )
parser.add_option('-p', 
				  dest="plot", 
                  action="store_true", 
				  default=False,
                  help='plots the resul: -p',
                  )

options, remainder = parser.parse_args()

if len(remainder) < 1:
	print 'The input file1 is missing!\n'
	parser.print_help()
	exit()

# passing the options
file1 = remainder[0]
cols1 = eval(options.cols1)

if len(remainder) > 1:
    file2 = remainder[1]
    cols2 = eval(options.cols2)
else:
    file2 = None
    cols2 = None

fileout = options.fileout
k = options.ndist
s0 = options.begin
sn = options.end
plot = options.plot

#--------------------------------------------------------------------------

def load_data(file1, cols1, file2=None, cols2=None):
    """Loads data in cols x y z (lon, lat, elem) from one or two 
	files, the latter is optional"""

    if file2 == None:
        print 'Loading data from file:\n%s' % file1
        return N.loadtxt(file1, usecols=cols1)
    else:
        print 'Loading data from files:\n%s\n%s' % (file1, file2)
        return N.loadtxt(file1, usecols=cols1), \
               N.loadtxt(file2, usecols=cols2)

#----------------------------------------------------------------------------

def save_data(fileout, COVS):
    N.savetxt(fileout, COVS[:,:3], fmt='%f %f %d', delimiter=' ')


# Searchs and calculates the covariances btw elements -----------------------
# Note: piece of intensive work written in C++/Blitz.
 
def autocov_xx(ELEM, s_k, ds):
    """AUTOCOVARIANCE between two arbitrary elements NOT azimuth 
	dependents"""

    nx = ELEM.shape[0]
    main_code = \
    """
    double xi, yi, li, xj, yj, lj, a, b, s_ij; 
    double sum_C_xx = 0, C_xx;
    int n = 0;
    py::tuple cov_xx(2);  // python tuple to return multiple values
    
    // Defines an interval [a,b) centered on s_k.
    a = s_k - ds/2.0;
    b = s_k + ds/2.0;
    
    for (int i = 0; i < nx; i++) {
        xi = ELEM(i,0);    // lon   (deg)
        yi = ELEM(i,1);    // lat   (deg)
        li = ELEM(i,2);    
    
        for (int j = i; j < nx; j++) {  // ATENTION: Autocov -> j = i !!!
            xj = ELEM(j,0);
            yj = ELEM(j,1);
            lj = ELEM(j,2);
    
            // Distance between pts P=(xi,yi) and Q=(xj,yj).
            s_ij = hypot(xj-xi, yj-yi);     // deg
    
	        // if the distance between pts belongs to s_k. 
            if (a <= s_ij && s_ij < b) {

	            // COVARIANCE between pts.
	            C_xx  = li * lj;

                // if the calculus returned valid numbers.
	            if (finite(C_xx)) {      
                    sum_C_xx += C_xx;
        	        n++;                    // number of products            
	            }
            }
        }
    }
    // Covariance value for all pts in a distance s_k.
    if (n > 0) {
        cov_xx[0] = sum_C_xx / n;
		cov_xx[1] = n;
	}
    else {
        cov_xx[0] = 0;
		cov_xx[1] = 0;
    }

    return_val = cov_xx; 
    """
    vars = ['ELEM', 's_k', 'ds', 'nx']
    return weave.inline(main_code, vars, type_converters=blitz)

#------------------------------------------------------------------------

def crosscov_xy(ELEMX, ELEMY, s_k, ds):
    """CROSSCOVARIANCE between two arbitrary elements NOT azimuth 
	dependents"""

    nx = ELEMX.shape[0]
    ny = ELEMY.shape[0]
    main_code = \
    """
    double xi, yi, li, xj, yj, lj, a, b, s_ij; 
    double sum_C_xy = 0, C_xy;
    int n = 0;
    py::tuple cov_xy(2);  // python tuple to return multiple values
    
    // Defines an interval [a,b) centered on s_k.
    a = s_k - ds/2.0;
    b = s_k + ds/2.0;
    
    for (int i = 0; i < nx; i++) {
        xi = ELEMX(i,0);    // lon   (deg)
        yi = ELEMX(i,1);    // lat   (deg)
        li = ELEMX(i,2);    
    
        for (int j = 0; j < ny; j++) {  // ATENTION: Crosscov -> j = 0 !!!
            xj = ELEMY(j,0);
            yj = ELEMY(j,1);
            lj = ELEMY(j,2);
    
            // Distance between pts P=(xi,yi) and Q=(xj,yj).
            s_ij = hypot(xj-xi, yj-yi);  // deg
    
            // if the distance between pts belongs to s_k. 
            if (a <= s_ij && s_ij < b) {

                // COVARIANCE between pts.
                C_xy  = li * lj;

                // if the calculus returned valid numbers.
                if (finite(C_xy)) {
                    sum_C_xy += C_xy;
                    n++;                  // number of products            
                }
            }
        }
    }
    // Covariance value for all pts in a distance s_k.
    if (n > 0) {
        cov_xy[0] = sum_C_xy / n;
        cov_xy[1] = n;
    }
    else {
        cov_xy[0] = 0;
        cov_xy[1] = 0;
    }

    return_val = cov_xy; 
    """
    vars = ['ELEMX', 'ELEMY', 's_k', 'ds', 'nx', 'ny']
    return weave.inline(main_code, vars, type_converters=blitz)

#----------------------------------------------------------------------------

def cov_distances(s0, sn, k, DIN1, DIN2=None):
    """Calculates covariances between diferent elements for k distances 
	in the interval [s0,sn) -> s_k = 0,1,2,..,k-1"""

    ds = (sn - s0) / k
    COVS = N.empty((k,5), 'float64')  # double

    # Autocov x-x 
    if DIN2 == None:
        for i in xrange(k):
            print 'Calculating auto-cov(x,x) for s_k = %s ...' % s0
            COVS[i,0] = s0 
            COVS[i,1], COVS[i,2] = autocov_xx(DIN1, s0, ds)
            s0 += ds

    # Crosscov x-y
    else:
        for i in xrange(k):
            print 'Calculating cross-cov(x,y) for s_k = %s ...' % s0
            COVS[i,0] = s0 
            COVS[i,1], COVS[i,2] = crosscov_xy(DIN1, DIN2, s0, ds)
            s0 += ds

    return COVS

#----------------------------------------------------------------------------

def plot_fig(COVS):

	P.plot(COVS[:,0], COVS[:,1], 'o')
	P.xlabel('spherical distance (deg)')
	P.ylabel('covariance')
	P.grid('True')
	P.show()

#----------------------------------------------------------------------------

def main():

    # One file -> Autocov
    if file2==None:
        DIN  = load_data(file1, cols1)
        COVS = cov_distances(s0, sn, k, DIN)

    # Two files -> Crosscov
    else:
        DIN1, DIN2  = load_data(file1, cols1, file2, cols2)
        COVS = cov_distances(s0, sn, k, DIN1, DIN2)

    save_data(fileout, COVS)
    print 'Output [dist,cov,nprod] -> ' + fileout

    if plot==True:
        plot_fig(COVS)



if __name__ == '__main__':
    main()
