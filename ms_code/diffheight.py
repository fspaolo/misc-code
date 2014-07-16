#!/usr/bin/env python

#    diffheight.py - Program to perform along-track differenced height.
#    
#    Input < lon (deg), lat (deg), SSH (m), time (sec)
#    Output > lon (deg), lat (deg), SDH (m), time (sec)
#    
#    Note:
#    (1) Not calculated gradients, between two pts with dt > GAP, are
#    marked with -1 in the time column. (2) Output -> N-1 lines
#    
#    Atention:
#    After gradient calculation must be applied the "clean.cpp" program,
#    in order to remove the lines marked with -1 in the output file.
#
#    Author:	Fernando Paolo 
#    Date:		jun/2008.
#    Usage:		python trackiff.py <filein> [fileout]

import pylab as P
import numpy as N
from sys import argv, exit
from os import system, path
from scipy import weave
from scipy.weave.converters import blitz


# Max time gap (seg) addmited for calculating diff.
GAP = 2000000    

#--------------------------------------------------------------------------

def check_arg(argv):
    """
    Check arguments and files.
    """
    # Check arguments.
    if len(argv) < 2:
        print 'Usage: python %s <filein> [fileout]' % argv[0]
        exit()
    elif len(argv) == 2:
        filein  = argv[1]
        fileout = 'diffheight.out'
    else:
        filein = argv[1]
        fileout = argv[2]
    # Check files.
    if not path.exists(filein):
        print 'Input file not found:' 
        print filein
        exit()
    return filein, fileout

#--------------------------------------------------------------------------

def lon_360to180(MAT):
    """
    Converts LON from 0/360 to -180/180 (only if LON > 180).
    Note: matrix columns must be LON, LAT, ...
    """
    nrow = MAT.shape[0]
    code = \
        """
	double lon;
	for (int i = 0; i < nrow; i++) {
        lon = MAT(i,0);
	    if (lon > 180)
	        MAT(i,0) = lon - 360.;
	}
	"""
    weave.inline(code, ['MAT', 'nrow'], type_converters=blitz)

#--------------------------------------------------------------------------

def lon_180to360(MAT):
    """
    Converts LON from -180/180 to 0/360 (only if LON < 0).
    Note: matrix columns must be LON, LAT, ...
    """
    nrow = MAT.shape[0]
    code = \
    """
	double lon;
	for (int i = 0; i < nrow; i++) {
        lon = MAT(i,0);
	    if (lon < 0)
	        MAT(i,0) = lon + 360.;
	}
	"""
    weave.inline(code, ['MAT', 'nrow'], type_converters=blitz)

#--------------------------------------------------------------------------
# Numerical differentiation, N-1 pts calculated.
#--------------------------------------------------------------------------
# Piece of intensive work written in C++/Blitz.

def diff_height(DIN, DOUT, GAP):

    """Calculates gradient and azimuth between two consecutive pts"""

    nrow = DIN.shape[0]
    code = \
    """
    double lon1, lon2, lat1, lat2, h1, h2, t1, t2;

    for (int i = 0; i < nrow-1; i++) {

        t1 = DIN(i,3); 
		t2 = DIN(i+1,3);
    
	    // if gap in the time is < GAP do:
        if ((t2 - t1) < GAP) {

            lon1 = DIN(i,0);    // deg 
            lat1 = DIN(i,1);    // deg
            lon2 = DIN(i+1,0);  // deg
            lat2 = DIN(i+1,1);  // deg
            h1   = DIN(i,2);    // m
	        h2   = DIN(i+1,2);  // m
    
	        // Calculates the difference between 2 consecutive pts (m).
            DOUT(i,0) = lon1;     // deg
            DOUT(i,1) = lat1;     // deg
            DOUT(i,2) = h2 - h1;  // m
            DOUT(i,3) = t1;       // sec
             
        }
        // else mark time with -1.
        else
            DOUT(i,3) = -1;
    }
    """
    vars = ['DIN', 'DOUT', 'GAP', 'nrow']
    weave.inline(code, vars, type_converters=blitz)
    
#--------------------------------------------------------------------------

def plot_fig(DIN, DOUT):
    """
    Plots h (SSH) and h2-h1 (SDH) for comparison.
    """
    P.subplot(211)
    P.plot(DIN[:,1], DIN[:,2])
    P.legend(["h (SSH)"])
    P.ylabel('height (m)')
    P.xlabel('latitude (deg)')
    
    P.subplot(212)
    P.plot(DOUT[:,1], DOUT[:,2], 'g')
    P.legend(['h2-h1 (SDH)'])
    P.ylabel('differenced height (m)')
    P.xlabel('latitude (deg)')
    
    P.savefig('ssh_sdh.eps')
    P.show()

#--------------------------------------------------------------------------

def main():

    # Checks arguments.
    filein, fileout = check_arg(argv)

    # Loads data into matrix.
    print 'Loading data ...'
    DIN = N.loadtxt(filein)
    #lon_360to180(DIN)  
    lon_180to360(DIN)  # transforms lon: -/+180 -> 0/360

    # Dimensions of output matrix.
    rows, cols = DIN.shape
    DOUT = N.empty((rows-1,cols), 'float64') # double = float64

    # Along-track differenced height.
    print 'Calculating differenced height ...'
    diff_height(DIN, DOUT, GAP)

    # Save file.
    N.savetxt(fileout, DOUT, fmt='%f', delimiter=' ')
    print 'Output -> ' + fileout

    # Plots image.
    #plot_fig(DIN, DOUT)


if __name__ == '__main__':
    main()
