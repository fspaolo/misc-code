#!/usr/bin/env python

#  colloc.py - Program to compute Free-air Anomaly or Geoid Height 
#  by the Least Squares Collocation (LSC) method, using sea surface 
#  gradients (SSG) and marine gravity anomalies (GRAV) as input data.
#  
#  Input < observations of type (1) 'SSG' or (2) 'SSG + GRAV'
#  Output > signal of type (a) GRAV or (b) GEOID
#  
#  Usage: python colloc.py -h
#
#  Author: Fernando Paolo 
#  Date: Jan/2009.


import numpy as N
import optparse
from sys import argv, exit
from scipy import weave
from scipy.weave.converters import blitz

# Editable part -----------------------------------------------------------

# minimum number of observations accepted per inversion cell 
# for signal calculation. ATTENTION: this number will depend on 
# the size of the cell
MIN_OBS_PER_CELL = 40  

# maximum value accepted for the predicted signal (to avoid anomalouss)
MAX_SIGNAL_VAL = 300.0

# if the prediction exceeds MAX_SIGNAL_VAL then is marked with:
ANOMALOUS_VAL = 0.0

# column (in the input file) of the standard deviation of each data type
COL_STD_SSG = 4
COL_STD_GRAV = 3

# Scan comand line arguments ----------------------------------------------

usage = "python %prog [options]"
epilog = "Units: lon (deg), lat (deg), esph_dist (deg), SSH (m), " + \
         "SSG (arcsec), grav (mGal), geoid (m), cov_ll (arcsec**2), " + \
		 "cov_mm (arcsec**2), cov_gg (mGal**2), cov_gl (mGal*arcsec), cov_nl (m*arcsec)"""
parser = optparse.OptionParser(usage=usage, epilog=epilog)
parser.add_option('-E', 
    	          dest='filessg',
                  default=None, 
                  help='data file with observations of type "SSG": -Essg.txt',
                  )
parser.add_option('-G', 
    	          dest='filegrav',
                  default=None, 
                  help='data file with observations of type "GRAV": -Ggrav.txt',
                  )
parser.add_option('-e',
    	          dest='colse',
                  default='(0,1,2,3,4)', 
                  help='cols to be loaded from FILESSG [lon,lat,ssg,azmth,err]: -e0,1,2,3,4',
                  )
parser.add_option('-g',
    	          dest='colsg',
                  default='(0,1,2,3)', 
                  help='cols to be loaded from FILEGRAV [lon,lat,grav,err]: -g0,1,2,3',
                  )
parser.add_option('-S', 
				  dest='signal',
				  default='g', 
				  help='signal to be computed of type "g" (grav) or "n" (geoid): -Sg',
				  )
parser.add_option('-A',
    	          dest='covll',
                  default='covll.txt', 
                  help='file with C_ll [deg,cov]: -Acovll.txt',
                  )
parser.add_option('-B',
    	          dest='covmm',
                  default='covmm.txt', 
                  help='file with C_mm [deg,cov]: -Bcovmm.txt',
                  )
parser.add_option('-C',
    	          dest='covgg',
                  default='covgg.txt', 
                  help='file with C_gg [deg,cov]: -Ccovgg.txt',
                  )
parser.add_option('-D',
    	          dest='covgl',
                  default='covgl.txt', 
                  help='file with C_gl [deg,cov]: -Dcovgl.txt',
                  )
parser.add_option('-F',
    	          dest='covnl',
                  default='covnl.txt', 
                  help='file with C_nl [deg,cov]: -Fcovnl.txt',
                  )
parser.add_option('-H',
    	          dest='covng',
                  default='covng.txt', 
                  help='file with C_ng [deg,cov]: -Hcovng.txt',
                  )
parser.add_option('-o',
    	          dest='fileout',
                  default='colloc.out', 
                  help='write output to FILEOUT [lon,lat,signal]: -ocolloc.out',
                  )
parser.add_option('-v',
    	          dest='varsig',
                  default=200.0, 
				  type='float',
                  help='variance of the signal to be computed: -v200.0',
                  )
parser.add_option('-d', 
    	          dest='diameter',
                  default=None, 
		          type='float',
                  help='diameter of circular cell used in the calculation [deg]: -d0.5',
                  )
parser.add_option('-l', 
    	          dest='side',
                  default=None, 
		          type='float',
                  help='side of square cell used in the calculation [deg]: -l0.5',
                  )
parser.add_option('-R', 
    	          dest='region',
                  default='(308,330,-6,8)', 
                  help='region to generate the grid [deg]: -Rwest,east,south,north',
                  )
parser.add_option('-I', 
    	          dest='resolut',
                  default=2, 
                  type='float',
                  help='final grid resolution [deg]: -I2',
                  )
parser.add_option('-s',
    	          dest='scale',
                  action='store_true',
                  default=False, 
                  help='use local (cell) scale factor for cov functions: -s',
                  )
options, remainder = parser.parse_args()

filessg = options.filessg
filegrav = options.filegrav

if filessg == None and filegrav == None:
    parser.print_help()
    exit()

colse = eval(options.colse)
colsg = eval(options.colsg)
signal = options.signal
fcovll = options.covll
fcovmm = options.covmm
fcovgg = options.covgg
fcovgl = options.covgl
fcovnl = options.covnl
fcovng = options.covng
fileout = options.fileout
var_sig = options.varsig
d = options.diameter
l = options.side
region = eval(options.region)
dx = options.resolut
dy = options.resolut
scale = options.scale

#--------------------------------------------------------------------------

def load_data(filessg, colse, filegrav=None, colsg=None):

    """Load data in cols x y z.. from one or two data files."""

    if filegrav==None:
        print 'loading data from file:\n%s' % filessg
        return N.loadtxt(filessg, usecols=colse)
    elif not filegrav==None:
        print 'loading data from files:\n%s\n%s' % (filessg, filegrav)
        return N.loadtxt(filessg, usecols=colse), \
               N.loadtxt(filegrav, usecols=colsg)

#--------------------------------------------------------------------------

def select_points(IN, OUT, lon, lat, d):

    """Selects the pts inside the circle pi*(d/2)**2 
	centering at (lon,lat) -> Circular inversion cell."""

    nrow = IN.shape[0]
    ncol = IN.shape[1]
    r = d/2.0

    functions = \
    """
	// Distance between two points (P and Q)

	double distance_pq(double lon_1, double lat_1, 
	                   double lon_2, double lat_2) {

			return hypot(lon_1 - lon_2, lat_1 - lat_2);
	}
    """
	
    main_code = \
    """
	int k = 0;
	double dist, x, y;

    for (int i = 0; i < nrow; i++) {
	    x = IN(i,0);  // lon
	    y = IN(i,1);  // lat

        dist = distance_pq(lon, lat, x, y);  // deg

	    if (dist <= r) {
	        for (int j = 0; j < ncol; j++){
		        OUT(k,j) = IN(i,j);
		    }
	        k++;
	   }
	}

	return_val = k;  // number of selected pts
	"""
    vars = ['IN', 'OUT', 'lon', 'lat', 'r', 'nrow', 'ncol']
    return weave.inline(main_code, vars, support_code=functions, 
	                    type_converters=blitz)

#--------------------------------------------------------------------------
def select_points2(IN, OUT, lon, lat, l):

    """Selects the pts inside the square l**2 centering 
	at (lon,lat) -> Square inversion cell"""

    nrow = IN.shape[0]
    ncol = IN.shape[1]
    r = l/2.0

    main_code = \
    """
	int k = 0;
	double x, y, xmin = lon, xmax = lon, ymin = lat, ymax = lat;

    xmin -= r;
	xmax += r;
	ymin -= r;
	ymax += r;

    for (int i = 0; i < nrow; i++) {
	    x = IN(i,0);  // lon
	    y = IN(i,1);  // lat

	    if ( xmin <= x && x <= xmax && ymin <= y && y <= ymax) {
	        for (int j = 0; j < ncol; j++){
		        OUT(k,j) = IN(i,j);
		    }
	        k++;
	   }
	}

	return_val = k;  // number of selected pts
	"""
    vars = ['IN', 'OUT', 'lon', 'lat', 'r', 'nrow', 'ncol']
    return weave.inline(main_code, vars, type_converters=blitz)

#--------------------------------------------------------------------------

def fills_cov_signal1(OBS, COVSL, C_sig, lon_grid, lat_grid, scale):

    """Fill vector of signal covariances: 
	
	    C_sig = [C_ge] or [C_ne]
	
	Finds covariances btw the signal (grav or geoid), in a grid point, 
	and the observations (SSG) to be used in the signal calculation 
	(inside a radius), by linear interpolation from tabulated covs: 
	a weighted average btw the two closest values is performed.

	OBS = SSG (sea surface gradients)
	COVSL = covariances btw signal and longitudinal component of SSG
	C_sig = vector of signal covariances
	scale = scale factor related to the cell (cov: global -> local)
	"""

    ne = OBS.shape[0] 
    ncov = COVSL.shape[0]

    functions = \
    """
    // Azimuth of direction defined by 2 pts (P and Q) --------------------
    // Note: the argument must be in DEGREES

    double azimuth_pq(double lon_1, double lat_1, 
                      double lon_2, double lat_2) {

	    const double PI = 3.1415926535897931;
	    const double R  = 6371007.1809;   // Earth's radius in m (WGS 84)
        double latm, dx, dy, azmth;

        double lon1 = lon_1 * (PI/180.);  // deg -> rad
        double lat1 = lat_1 * (PI/180.);  // rad
        double lon2 = lon_2 * (PI/180.);  // rad
        double lat2 = lat_2 * (PI/180.);  // rad

        latm = (lat1 + lat2) / 2.0;          // rad
        dx = R * cos(latm) * (lon2 - lon1);  // rad
        dy = R * (lat2 - lat1);              // rad
		azmth = atan2(dx, dy); 

		// For getting azmth > 0 always.
		if (azmth < 0)
            azmth += 2*PI;  // rad

        return azmth;       // azimuth (> 0) in rad
    }

	// Distance between two point (P and Q) -------------------------------

	double distance_pq(double lon_1, double lat_1, 
	                   double lon_2, double lat_2) {

			return hypot(lon_1 - lon_2, lat_1 - lat_2);
	}
	"""

    main_code = \
    """
	using namespace std;
    double lon_e, lat_e, azmth_e, theta_e, azmth_pq, dist_pq;
	double dist1, dist2, cov1, cov2, w1, w2, C_sl;
	double s = scale;

    for (int i = 0; i < ne; i++) {

        lon_e = OBS(i,0);    // SSG lon (deg)
        lat_e = OBS(i,1);    // SSG lat (deg)
		azmth_e = OBS(i,3);  // SSG azimuth (rad)
     
	    // azimuth between pts P (grid-signal) and Q (observations)
	    azmth_pq = azimuth_pq(lon_grid, lat_grid, lon_e, lat_e);  // rad

        // angle of ssg related to PQ direction
        theta_e = azmth_e - azmth_pq;       // rad

        // distance btw pts P (grid-signal) and Q (observation)
        dist_pq = distance_pq(lon_grid, lat_grid, lon_e, lat_e);  // deg
     
        // finds a cov value according to distance btw pts
        for (int j = 0; j < ncov-1; j++) {

            dist1 = COVSL(j,0);    // tabulated distance
            cov1 = COVSL(j,1);     // tabulated covariance 
            dist2 = COVSL(j+1,0);
            cov2 = COVSL(j+1,1);     

			// a) if there is an exact tabulated distance no average is needed
			if (dist_pq == dist1) {
				C_sl = cov1;
        	    break;
            }
			// b) if it is btw two values a weighted average is performed
            else if (dist1 < dist_pq && dist_pq <= dist2) {
                w1 = (dist2 - dist_pq)/(dist2 - dist1);   // weight cov1 
                w2 = 1.0 - w1;                            // weight cov2
                C_sl = w1 * cov1 + w2 * cov2;             // weighted average
        	    break;
            }
			// c) if there isn't tabulated values for this distance in the file
            else if (j == ncov-2) {
                cout << "Radius to large, no C_sl for: " << dist_pq << " deg, "
                     << "filling with 0.0" << endl;
        	    C_sl = 0.0;
            }
        }

		// calculates C_ge or C_ne ----------------------------------------

        C_sig(i) = (-cos(theta_e) * C_sl) * s;
    }
    """
    vars = ['OBS', 'COVSL', 'C_sig', 'lon_grid', 'lat_grid', 'ne', \
	        'ncov', 'scale']
    weave.inline(main_code, vars, support_code=functions, type_converters=blitz)

#--------------------------------------------------------------------------

def fills_cov_signal2(OBS, COVSL, COVSG, C_sig, lon_grid, lat_grid, 
                      ne, ng, scale):

    """Fills vector of signal covariances: 
	
	    C_sig = [C_ge C_gg] or [C_ne C_ng]
	
	Finds covariances btw the signal (grav or geoid), in a grid point, 
	and the observations (SSG and GRAV) to be used in the signal 
	calculation (inside a radius), by linear interpolation from tabulated 
	covs: a weighted average btw the two closest values is performed.

	OBS = SSG + GRAV (sea surface gradients + gravity anomalies)
	COVSL = covariances btw signal and longitudinal component of SSG
	COVSG = covariances btw signal and gravity anomaly (GRAV)
	C_sig = vector of signal covariances
	ne = number of observations SSG
	ng = number of observations GRAV
	scale = scale factor related to the cell (cov: global -> local)
	"""

    ncsl = COVSL.shape[0]
    ncsg = COVSG.shape[0]

    functions = \
    """
    // Azimuth of direction defined by 2 pts (P and Q) --------------------
    // Note: the argument must be in DEGREES

    double azimuth_pq(double lon_1, double lat_1, 
                      double lon_2, double lat_2) {

	    const double PI = 3.1415926535897931;
	    const double R  = 6371007.1809;   // Earth's radius in m (WGS 84)
        double latm, dx, dy, azmth;

        double lon1 = lon_1 * (PI/180.);  // deg -> rad
        double lat1 = lat_1 * (PI/180.);  // rad
        double lon2 = lon_2 * (PI/180.);  // rad
        double lat2 = lat_2 * (PI/180.);  // rad

        latm = (lat1 + lat2) / 2.0;          // rad
        dx = R * cos(latm) * (lon2 - lon1);  // rad
        dy = R * (lat2 - lat1);              // rad
		azmth = atan2(dx, dy); 

		// For getting azmth > 0 always.
		if (azmth < 0)
            azmth += 2*PI;  // rad

        return azmth;       // azimuth (> 0) in rad
    }

	// Distance between two point (P and Q) -------------------------------

	double distance_pq(double lon_1, double lat_1, 
	                   double lon_2, double lat_2) {

			return hypot(lon_1 - lon_2, lat_1 - lat_2);
	}
	"""

    main_code = \
    """
    using namespace std;
    double lon_e, lat_e, azmth_e, theta_e, azmth_pq, dist_pq;
	double lon_grav, lat_grav;
	double dist1, dist2, cov1, cov2, w1, w2, C_sl, C_sg;
	double s = scale;

    // (1) fills first part of C_sig with COVSL ---------------------------
    // from [0] to [ne-1] with C_se

    for (int i = 0; i < ne; i++) {  

        lon_e = OBS(i,0);         // SSG lon (deg)
        lat_e = OBS(i,1);         // SSG lat (deg)
		azmth_e = OBS(i,3);       // SSG azimuth (rad)
     
	    // azimuth between pts P (grid-signal) and Q (observations)
	    azmth_pq = azimuth_pq(lon_grid, lat_grid, lon_e, lat_e);  // rad

        // angle of SSG at Q related to PQ direction
        theta_e = azmth_e - azmth_pq;       // rad

        // distance btw pts P (grid-signal) and Q (observation)
        dist_pq = distance_pq(lon_grid, lat_grid, lon_e, lat_e);  // deg
     
        // finds a cov value according to distance btw pts
        for (int j = 0; j < ncsl-1; j++) {

            dist1 = COVSL(j,0);    // tabulated distance
            cov1 = COVSL(j,1);     // tabulated covariance 
            dist2 = COVSL(j+1,0);
            cov2 = COVSL(j+1,1);     

			// a) if there is an exact tabulated distance no average is needed
			if (dist_pq == dist1) {
				C_sl = cov1;
        	    break;
            }
			// b) if it is btw two values a weighted average is performed
            else if (dist1 < dist_pq && dist_pq <= dist2) {
                w1 = (dist2 - dist_pq)/(dist2 - dist1);   // weight cov1 
                w2 = 1.0 - w1;                            // weight cov2
                C_sl = w1 * cov1 + w2 * cov2;             // weighted average
        	    break;
            }
			// c) if there isn't tabulated values for this distance in the file
            else if (j == ncsl-2) {
                cout << "Radius to large, no C_sl for: " << dist_pq << " deg, "
                     << "filling with 0.0" << endl;
        	    C_sl = 0.0;
            }
        }

		// calculates C_ge or C_ne ----------------------------------------

        C_sig(i) = (-cos(theta_e) * C_sl) * s;
    }

    // (2) fills second part of C_sig with COVSG --------------------------
    // from [ne] to [ne+ng-1] with C_sg

    for (int i = ne; i < ne+ng; i++) {  

        lon_grav = OBS(i,0);  // lon GRAV at Q
        lat_grav = OBS(i,1);  // lat GRAV at Q

        // distance btw pts P (grid-signal) and Q (observation)
        dist_pq = distance_pq(lon_grid, lat_grid, lon_grav, lat_grav);  // deg
     
        // finds a cov value according to distance btw pts
        for (int j = 0; j < ncsg-1; j++) {

            dist1 = COVSG(j,0);    // tabulated distance
            cov1 = COVSG(j,1);     // tabulated covariance 
            dist2 = COVSG(j+1,0);
            cov2 = COVSG(j+1,1);     

			// a) if there is an exact tabulated distance no average is needed
			if (dist_pq == dist1) {
				C_sg = cov1;
        	    break;
            }
			// b) if it is btw two values a weighted average is performed
            else if (dist1 < dist_pq && dist_pq <= dist2) {
                w1 = (dist2 - dist_pq)/(dist2 - dist1);   // weight cov1 
                w2 = 1.0 - w1;                            // weight cov2
                C_sg = w1 * cov1 + w2 * cov2;             // weighted average
        	    break;
            }
			// c) if there isn't tabulated values for this distance in the file
            else if (j == ncsg-2) {
                cout << "Radius to large, no C_sg for: " << dist_pq << " deg, "
                     << "filling with 0.0" << endl;
        	    C_sg = 0.0;
            }
        }

		// C_gg or C_ng ---------------------------------------------------

		C_sig(i) = C_sg * s;
    }
    """
    vars = ['OBS', 'COVSL', 'COVSG', 'C_sig', 'lon_grid', 'lat_grid', \
	        'ne', 'ng', 'ncsl', 'ncsg', 'scale']
    weave.inline(main_code, vars, support_code=functions, type_converters=blitz)

#--------------------------------------------------------------------------

def fills_cov_observ1(OBS, COVLL, COVMM, C_OBS, scale):

    """Fills matrix of observations covariances: 
	
	    C_OBS = [C_ee + D_e]
	
	Finds covariances btw the observations (SSG), in a radius, to be 
	used in the signal (grid point) calculation, by linear interpolation 
	from tabulated covs: a weighted average btw the two closest values 
	is performed.

	The error variance is added to the diagonal elements.

	OBS = SSG (sea surface gradients inside a radius)
	COVLL = covariances btw longitudinal components of SSG
	COVMM = covariances btw transversal components of SSG
	C_OBS = matrix of observations covariances
	scale = scale factor related to the cell (cov: global -> local)
	"""

    ne = OBS.shape[0] 
    ncll = COVLL.shape[0]
    ncmm = COVMM.shape[0]

    functions = \
    """
    // Azimuth of direction defined by 2 pts (P and Q) --------------------
    // Note: the argument must be in DEGREES

    double azimuth_pq(double lon_1, double lat_1, 
                      double lon_2, double lat_2) {

	    const double PI = 3.1415926535897931;
	    const double R  = 6371007.1809;   // Earth's radius in m (WGS 84)
        double latm, dx, dy, azmth;

        double lon1 = lon_1 * (PI/180.);  // deg -> rad
        double lat1 = lat_1 * (PI/180.);  // rad
        double lon2 = lon_2 * (PI/180.);  // rad
        double lat2 = lat_2 * (PI/180.);  // rad

        latm = (lat1 + lat2) / 2.0;          // rad
        dx = R * cos(latm) * (lon2 - lon1);  // rad
        dy = R * (lat2 - lat1);              // rad
		azmth = atan2(dx, dy); 

		// For getting azmth > 0 always.
		if (azmth < 0)
            azmth += 2*PI;  // rad

        return azmth;       // azimuth (> 0) in rad
    }

	// Distance between two point (P and Q) -------------------------------

	double distance_pq(double lon_1, double lat_1, 
	                   double lon_2, double lat_2) {

			return hypot(lon_1 - lon_2, lat_1 - lat_2);
	}
	"""

    main_code = \
    """
    using namespace std;
    double lon_p, lat_p, azmth_p, theta_p, lon_q, lat_q, azmth_q, theta_q; 
	double azmth_pq, dist_pq, C_ll, C_mm; 
	double dist1, dist2, cov1, cov2, w1, w2;

	double s = scale, std;
	double URAD_TO_ARCSEC = 0.20626480599999999;


    for (int i = 0; i < ne; i++) {

        lon_p = OBS(i,0);    // lon SSG at P (deg)
        lat_p = OBS(i,1);    // lat SSG at P (deg)
		azmth_p = OBS(i,3);  // azimuth SSG at P (rad)

        for (int j = 0; j < ne; j++) {

            lon_q = OBS(j,0);    // lon SSG at Q
            lat_q = OBS(j,1);    // lat SSG at Q
		    azmth_q = OBS(j,3);  // azimuth SSG at Q

	        // azimuth between points P and Q (SSGs)
	        azmth_pq = azimuth_pq(lon_p, lat_p, lon_q, lat_q);  // rad

            // Angles of SSGs at P and Q related to PQ direction
            theta_p = azmth_p - azmth_pq;  // rad
            theta_q = azmth_q - azmth_pq;  // rad
            
            // distance btw SSGs at P=(lon_p,lat_p) and Q=(lon_q,lat_q)
            dist_pq = distance_pq(lon_p, lat_p, lon_q, lat_q);  // deg
            
            // finds a cov value according to distance btw pts:

            // for C_ll ---------------------------------------------------

            for (int k = 0; k < ncll-1; k++) {

                dist1 = COVLL(k,0);    // tabulated distance
                cov1 = COVLL(k,1);     // tabulated covariance 
                dist2 = COVLL(k+1,0);
                cov2 = COVLL(k+1,1);     

		        // a) if there is an exact tabulated distance no average is needed
		        if (dist_pq == dist1) {
		        	C_ll = cov1;
                    break;
                }
		        // b) if it is btw two values a weighted average is performed
                else if (dist1 < dist_pq && dist_pq <= dist2) {
                    w1 = (dist2 - dist_pq)/(dist2 - dist1);  // weight cov1 
                    w2 = 1.0 - w1;                           // weight cov2
                    C_ll = w1 * cov1 + w2 * cov2;            // weighted average
                    break;
                }
		        // c) if there isn't tabulated values for this distance in the file
                else if (k == ncll-2) {
                    cout << "Radius to large, no C_ll for: " << dist_pq << " deg, "
                         << "filling with 0.0" << endl;
                    C_ll = 0.0;
                }
            }

            // for C_mm ---------------------------------------------------

            for (int k = 0; k < ncmm-1; k++) {

                dist1 = COVMM(k,0);    // tabulated distance
                cov1 = COVMM(k,1);     // tabulated covariance 
                dist2 = COVMM(k+1,0);
                cov2 = COVMM(k+1,1);     

		        // a) if there is an exact tabulated distance no average is needed
		        if (dist_pq == dist1) {
		        	C_mm = cov1;
                    break;
                }
		        // b) if it is btw two values a weighted average is performed
                else if (dist1 < dist_pq && dist_pq <= dist2) {
                    w1 = (dist2 - dist_pq)/(dist2 - dist1);  // weight cov1 
                    w2 = 1.0 - w1;                           // weight cov2
                    C_mm = w1 * cov1 + w2 * cov2;            // weighted average
                    break;
                }
		        // c) if there isn't tabulated values for this distance in the file
                else if (k == ncmm-2) {
                    cout << "Radius to large, no C_mm for: " << dist_pq << " deg, "
                         << "filling with 0.0" << endl;
                    C_mm = 0.0;
                }
            }

			// calculates C_ee --------------------------------------------

            C_OBS(i,j) = (C_ll * cos(theta_p) * cos(theta_q)
                         + C_mm * sin(theta_p) * sin(theta_q)) * s;

            // Adds the error variance to the diagonal element ------------

            if (i == j && C_OBS(i,j) != 0) {                // TER CERTEZA !!!
                
				std = OBS(i,COL_STD_SSG) * URAD_TO_ARCSEC;  // urad > arcsec
                C_OBS(i,j) += std * std;                    // D_e (arcsec**2)
            }
        }
    }
    """
    vars = ['OBS', 'COVLL', 'COVMM', 'C_OBS', 'ne', 'ncll', 'ncmm', 'scale',
	        'COL_STD_SSG']
    weave.inline(main_code, vars, support_code=functions, type_converters=blitz)

#--------------------------------------------------------------------------

def fills_cov_observ2(OBS, COVLL, COVMM, COVGG, COVGL, C_OBS, ne, ng, scale):

    """Fills matrix of observations covariances: 
	
        C_OBS = [C_ee + D_e  C_eg]
                [C_ge  C_gg + D_g]
	
	Finds covariances btw the observations (SSG + GRAV), in a radius, 
	to be used in the signal (grid point) calculation, by linear 
	interpolation from tabulated covs: a weighted average btw the 
	two closest values is performed.

	The error variance is added to the diagonal elements.

	OBS = SSG (sea surface gradients inside a radius)
	COVLL = covariances btw longitudinal components of SSG
	COVMM = covariances btw transversal components of SSG
	COVGG = covariances btw gravity anomalies GRAV
	COVGL = covariances btw gravity anomaly and longitudinal comp of SSG
	C_OBS = matrix of observations covariances
	ne = number of SSGs
	ng = number of GRAVs
	scale = scale factor related to the cell (cov: global -> local)
	"""

    ncll = COVLL.shape[0]
    ncmm = COVMM.shape[0]
    ncgg = COVGG.shape[0]
    ncgl = COVGL.shape[0]

    functions = \
    """
    // Azimuth of direction defined by 2 pts (P and Q) --------------------
    // Note: the argument must be in DEGREES

    double azimuth_pq(double lon_1, double lat_1, 
                      double lon_2, double lat_2) {

	    const double PI = 3.1415926535897931;
	    const double R  = 6371007.1809;   // Earth's radius in m (WGS 84)
        double latm, dx, dy, azmth;

        double lon1 = lon_1 * (PI/180.);  // deg -> rad
        double lat1 = lat_1 * (PI/180.);  // rad
        double lon2 = lon_2 * (PI/180.);  // rad
        double lat2 = lat_2 * (PI/180.);  // rad

        latm = (lat1 + lat2) / 2.0;          // rad
        dx = R * cos(latm) * (lon2 - lon1);  // rad
        dy = R * (lat2 - lat1);              // rad
		azmth = atan2(dx, dy); 

		// For getting azmth > 0 always.
		if (azmth < 0)
            azmth += 2*PI;  // rad

        return azmth;       // azimuth (> 0) in rad
    }

	// Distance between two point (P and Q) -------------------------------

	double distance_pq(double lon_1, double lat_1, 
	                   double lon_2, double lat_2) {

			return hypot(lon_1 - lon_2, lat_1 - lat_2);
	}
	"""

    main_code = \
    """
    using namespace std;
    double lon_p, lat_p, azmth_p, theta_p, lon_q, lat_q, azmth_q, theta_q; 
	double lon_e, lat_e, azmth_e, theta_e, lon_g, lat_g;
	double azmth_pq, dist_pq, C_ll, C_mm, C_gg, C_gl, C_ee, C_ge, C_eg; 
	double dist1, dist2, cov1, cov2, w1, w2;

	double s = scale, std;
	double URAD_TO_ARCSEC = 0.20626480599999999;


    // (1) fills 1st part of C_OBS with COVLL and COVMM -------------------
	// from (0,0) to (ne-1,ne-1) with C_ee + D_e

    for (int i = 0; i < ne; i++) {

        lon_p = OBS(i,0);    // lon SSG at P (deg)
        lat_p = OBS(i,1);    // lat SSG at P (deg)
		azmth_p = OBS(i,3);  // azimuth SSG at P (rad)

        for (int j = 0; j < ne; j++) {

            lon_q = OBS(j,0);    // lon SSG at Q
            lat_q = OBS(j,1);    // lat SSG at Q
		    azmth_q = OBS(j,3);  // azimuth SSG at Q

	        // azimuth between points P and Q (SSGs)
	        azmth_pq = azimuth_pq(lon_p, lat_p, lon_q, lat_q);  // rad

            // Angles of SSGs at P and Q related to PQ direction
            theta_p = azmth_p - azmth_pq;  // rad
            theta_q = azmth_q - azmth_pq;  // rad
            
            // distance btw SSGs at P=(lon_p,lat_p) and Q=(lon_q,lat_q)
            dist_pq = distance_pq(lon_p, lat_p, lon_q, lat_q);  // deg
            
            // finds a cov value according to distance btw pts:

            // for C_ll ---------------------------------------------------

            for (int k = 0; k < ncll-1; k++) {

                dist1 = COVLL(k,0);    // tabulated distance
                cov1 = COVLL(k,1);     // tabulated covariance 
                dist2 = COVLL(k+1,0);
                cov2 = COVLL(k+1,1);     

		        // a) if there is an exact tabulated distance no average is needed
		        if (dist_pq == dist1) {
		        	C_ll = cov1;
                    break;
                }
		        // b) if it is btw two values a weighted average is performed
                else if (dist1 < dist_pq && dist_pq <= dist2) {
                    w1 = (dist2 - dist_pq)/(dist2 - dist1);  // weight cov1 
                    w2 = 1.0 - w1;                           // weight cov2
                    C_ll = w1 * cov1 + w2 * cov2;            // weighted average
                    break;
                }
		        // c) if there isn't tabulated values for this distance in the file
                else if (k == ncll-2) {
                    cout << "Radius to large, no C_ll for: " << dist_pq << " deg, "
                         << "filling with 0.0" << endl;
                    C_ll = 0.0;
                }
            }

            // for C_mm ---------------------------------------------------

            for (int k = 0; k < ncmm-1; k++) {

                dist1 = COVMM(k,0);    // tabulated distance
                cov1 = COVMM(k,1);     // tabulated covariance 
                dist2 = COVMM(k+1,0);
                cov2 = COVMM(k+1,1);     

		        // a) if there is an exact tabulated distance no average is needed
		        if (dist_pq == dist1) {
		        	C_mm = cov1;
                    break;
                }
		        // b) if it is btw two values a weighted average is performed
                else if (dist1 < dist_pq && dist_pq <= dist2) {
                    w1 = (dist2 - dist_pq)/(dist2 - dist1);  // weight cov1 
                    w2 = 1.0 - w1;                           // weight cov2
                    C_mm = w1 * cov1 + w2 * cov2;            // weighted average
                    break;
                }
		        // c) if there isn't tabulated values for this distance in the file
                else if (k == ncmm-2) {
                    cout << "Radius to large, no C_mm for: " << dist_pq << " deg, "
                         << "filling with 0.0" << endl;
                    C_mm = 0.0;
                }
            }

			// calculates C_ee --------------------------------------------

            C_OBS(i,j) = (C_ll * cos(theta_p) * cos(theta_q)     
                         + C_mm * sin(theta_p) * sin(theta_q)) * s;

            // Adds the error variance to the diagonal element ------------

            if (i == j && C_OBS(i,j) != 0) {                // TER CERTEZA !!!
                
				std = OBS(i,COL_STD_SSG) * URAD_TO_ARCSEC;  // urad > arcsec
                C_OBS(i,j) += std * std;                    // D_e (arcsec**2)
            }
        }
    }

    // (2) fills 2nd part of C_OBS with COVGL -----------------------------
	// from (0,ne) to (ne-1,ne+ng-1) with C_eg

    for (int i = 0; i < ne; i++) {

        lon_e = OBS(i,0);    // lon SSG at P (deg)
        lat_e = OBS(i,1);    // lat SSG at P (deg)
		azmth_e = OBS(i,3);  // azimuth SSG at P (rad)

        for (int j = ne; j < ne+ng; j++) {

            lon_g = OBS(j,0);    // lon GRAV at Q
            lat_g = OBS(j,1);    // lat GRAV at Q

	        // azimuth of PQ direction defined by pts P (SSG) and Q (GRAV)
	        azmth_pq = azimuth_pq(lon_e, lat_e, lon_g, lat_g);  // rad

            // Angle of SSG at P related to PQ direction
            theta_e = azmth_e - azmth_pq;  // rad
            
            // distance btw SSG at P=(lon_e,lat_e) and GRAV at Q=(lon_g,lat_g)
            dist_pq = distance_pq(lon_e, lat_e, lon_g, lat_g);  // deg
            
            // finds a cov value according to distance btw pts:

            // for C_gl ---------------------------------------------------

            for (int k = 0; k < ncgl-1; k++) {

                dist1 = COVGL(k,0);    // tabulated distance
                cov1 = COVGL(k,1);     // tabulated covariance 
                dist2 = COVGL(k+1,0);
                cov2 = COVGL(k+1,1);     

		        // a) if there is an exact tabulated distance no average is needed
		        if (dist_pq == dist1) {
		        	C_gl = cov1;
                    break;
                }
		        // b) if it is btw two values a weighted average is performed
                else if (dist1 < dist_pq && dist_pq <= dist2) {
                    w1 = (dist2 - dist_pq)/(dist2 - dist1);  // weight cov1 
                    w2 = 1.0 - w1;                           // weight cov2
                    C_gl = w1 * cov1 + w2 * cov2;            // weighted average
                    break;
                }
		        // c) if there isn't tabulated values for this distance in the file
                else if (k == ncgl-2) {
                    cout << "Radius to large, no C_gl for: " << dist_pq << " deg, "
                         << "filling with 0.0" << endl;
                    C_gl = 0.0;
                }
            }

			// calculates C_eg --------------------------------------------

            C_OBS(i,j) = (cos(theta_e) * C_gl) * s;
        }
    }

    // (3) fills 3th part of C_OBS with COVGL -----------------------------
	// from (ne,0) to (ne+ng-1,ne-1) with C_ge

    for (int i = ne; i < ne+ng; i++) {

        lon_g = OBS(i,0);    // lon GRAV at P
        lat_g = OBS(i,1);    // lat GRAV at P

        for (int j = 0; j < ne; j++) {

            lon_e = OBS(j,0);    // lon SSG at Q (deg)
            lat_e = OBS(j,1);    // lat SSG at Q (deg)
	        azmth_e = OBS(j,3);  // azimuth SSG at Q (rad)

	        // azimuth of PQ direction defined by pts P (GRAV) and Q (SSG)
	        azmth_pq = azimuth_pq(lon_g, lat_g, lon_e, lat_e);  // rad

            // Angle of SSG at Q related to PQ direction
            theta_e = azmth_e - azmth_pq;  // rad
            
            // distance btw GRAV at P=(lon_g,lat_g) and SSG at Q=(lon_e,lat_e)
            dist_pq = distance_pq(lon_g, lat_g, lon_e, lat_e);  // deg
            
            // finds a cov value according to distance btw pts:

            // for C_gl ---------------------------------------------------

            for (int k = 0; k < ncgl-1; k++) {

                dist1 = COVGL(k,0);    // tabulated distance
                cov1 = COVGL(k,1);     // tabulated covariance 
                dist2 = COVGL(k+1,0);
                cov2 = COVGL(k+1,1);     

		        // a) if there is an exact tabulated distance no average is needed
		        if (dist_pq == dist1) {
		        	C_gl = cov1;
                    break;
                }
		        // b) if it is btw two values a weighted average is performed
                else if (dist1 < dist_pq && dist_pq <= dist2) {
                    w1 = (dist2 - dist_pq)/(dist2 - dist1);  // weight cov1 
                    w2 = 1.0 - w1;                           // weight cov2
                    C_gl = w1 * cov1 + w2 * cov2;            // weighted average
                    break;
                }
		        // c) if there isn't tabulated values for this distance in the file
                else if (k == ncgl-2) {
                    cout << "Radius to large, no C_gl for: " << dist_pq << " deg, "
                         << "filling with 0.0" << endl;
                    C_gl = 0.0;
                }
            }

			// calculates C_ge --------------------------------------------

            C_OBS(i,j) = (-cos(theta_e) * C_gl) * s;
        }
    }

    // (4) fills 4th part of C_OBS with COVGG -----------------------------
	// from (ne,ne) to (ne+ng-1,ne+ng-1) with C_gg + D_g

    for (int i = ne; i < ne+ng; i++) {

        lon_p = OBS(i,0);    // lon GRAV at P (deg)
        lat_p = OBS(i,1);    // lat GRAV at P (deg)

        for (int j = ne; j < ne+ng; j++) {

            lon_q = OBS(j,0);    // lon GRAV at Q
            lat_q = OBS(j,1);    // lat GRAV at Q

            // distance btw GRAVs at P=(lon_p,lat_p) and Q=(lon_q,lat_q)
            dist_pq = distance_pq(lon_p, lat_p, lon_q, lat_q);  // deg
            
            // finds a cov value according to distance btw pts:

            // for C_gg ---------------------------------------------------

            for (int k = 0; k < ncgg-1; k++) {

                dist1 = COVGG(k,0);    // tabulated distance
                cov1 = COVGG(k,1);     // tabulated covariance 
                dist2 = COVGG(k+1,0);
                cov2 = COVGG(k+1,1);     

		        // a) if there is an exact tabulated distance no average is needed
		        if (dist_pq == dist1) {
		        	C_gg = cov1;
                    break;
                }
		        // b) if it is btw two values a weighted average is performed
                else if (dist1 < dist_pq && dist_pq <= dist2) {
                    w1 = (dist2 - dist_pq)/(dist2 - dist1);  // weight cov1 
                    w2 = 1.0 - w1;                           // weight cov2
                    C_gg = w1 * cov1 + w2 * cov2;            // weighted average
                    break;
                }
		        // c) if there isn't tabulated values for this distance in the file
                else if (k == ncll-2) {
                    cout << "Radius to large, no C_gg for: " << dist_pq << " deg, "
                         << "filling with 0.0" << endl;
                    C_gg = 0.0;
                }
            }

			// C_gg -------------------------------------------------------

            C_OBS(i,j) = C_gg * s;

            // Adds the error variance to the diagonal element ------------
            if (i == j && C_OBS(i,j) != 0) {                // TER CERTEZA !!!
                
				std = OBS(i,COL_STD_GRAV);                  // mGal
                C_OBS(i,j) += std * std;                    // D_g (mGal**2)
            }
        }
    }
    """
    vars = ['OBS', 'COVLL', 'COVMM', 'COVGG', 'COVGL', 'C_OBS', 'ne', 'ng', \
	        'ncll', 'ncmm', 'ncgg', 'ncgl', 'scale', 'COL_STD_SSG', 'COL_STD_GRAV']
    weave.inline(main_code, vars, support_code=functions, type_converters=blitz)

#--------------------------------------------------------------------------

def LSC_solver1(C_sig, C_OBS, obs, var_sig):

    """solves the Least Squares Collocation system: 

        signal = C_sig * C_OBS_inv * obs
        error = variance - C_sig * C_OBS_inv * C_sig_trasnp

    using the Hwang and Parsons (1995) algorithm:
    
        signal = b.T * y
        error = variance - b.T * b
		
    for one single point.
    """

    # signal calculation 
    L = N.linalg.cholesky(C_OBS)     # cholesky decomposition = L_MAT
    L_i = N.linalg.inv(L)            # L_MATRIX invertion
    b = N.dot(L_i, C_sig.T)          # L_MAT_inv * sig_VEC_transp = b_VEC
    b_t = b.T                        # b_VEC transpost
    y = N.dot(L_i, obs)              # L_MAT_inv * obs_VEC = y_VEC
    signal = N.dot(b_t, y)           # b_VEC_transp * y_VEC
    
    # error calculation
    error = var_sig - N.dot(b_t, b)  # var_ESC - b_VEC_transp * b_VEC

    return signal, error

#--------------------------------------------------------------------------

def LSC_solver2(C_sig, C_OBS, obs, var_sig):

    """solves the Least Squares Collocation system: 

        signal = C_sig * C_OBS_inv * obs
        error = variance - C_sig * C_OBS_inv * C_sig_transp

    using conventional matrix invertion, for one single point.
    """

    # signal calculation 
    C_OBS_i = N.linalg.inv(C_OBS)               # matrix invertion
    signal = N.dot(N.dot(C_sig, C_OBS_i), obs)  # dot product

    # error calculation
    C_sig_t = C_sig.T                           # vector transpost
    error = var_sig - N.dot(N.dot(C_sig, C_OBS_i), C_sig_t)

    return signal, error

#--------------------------------------------------------------------------

def scale_factor(obs, COVLL, ne, COVGG=False, ng=0):

    """calculates the local scale factor for each cell to scale the
	covariance functions in the inversion procedure.

    obs = observations
	COVLL = cov(l,l)
	ne = number of SSGs
	COVGG = cov(g,g)
	ng = number of GRAVs
	"""

    convert_factor = 23.504430595412476       # asec**2 -> urad**2
    #convert_factor = 1.0/23.504430595412476  # urad**2 -> asec**2
    
    # ratio btw estimated (local) and modeled (global) variances

    ## a = ratio for SSGs
    local_var_e = N.var(obs[:ne])	 
    global_var_e = COVLL[0,1] * convert_factor  
    a = local_var_e / global_var_e            # adimensional

    ## b = ratio for GRAVs
    if not ng == 0:
        local_var_g = N.var(obs[ne:ng])	 
        global_var_g = COVGG[0,1]
        b = local_var_g / global_var_g
    else:
        b = 0

    # scale factor related to the cell
    scale = (ne * a + ng * b) / (ne + ng)

    # to ensure the covariance wont be increased: 0 <= s <= 1
    if scale <= 1.0:
        s = scale
    else:
        s = 1.0

    return s


#--------------------------------------------------------------------------

def main():

    ### (1) observations: SSG
    if filegrav == None:             
        DATAE = load_data(filessg, colse) 
        ne = DATAE.shape[0]
        AUXE = N.empty((ne, 5), 'float64')
        print 'observations: SSG'

        # a) signal: GRAV | covs: C_ll, C_mm, C_gl
        if signal == 'g':
            COVLL = N.loadtxt(fcovll)
            COVMM = N.loadtxt(fcovmm)
            COVGL = N.loadtxt(fcovgl)
            print 'signal: GRAV'
            print 'covariances: C_ll, C_mm, C_gl'
        # b) signal: GEOID | covs: C_ll, C_mm, C_nl
        elif signal == 'n':
            COVLL = N.loadtxt(fcovll)
            COVMM = N.loadtxt(fcovmm)
            COVNL = N.loadtxt(fcovnl)
            print 'signal: GEOID'
            print 'covariances: C_ll, C_mm, C_nl'
        else:
            print 'Error with signal choice: -s[g|n]'
            exit()

    ### (2) observations: SSG + GRAV
    else: 
        DATAE, DATAG = load_data(filessg, colse, filegrav, colsg) 
        ne = DATAE.shape[0]
        ng = DATAG.shape[0]
        AUXE = N.empty((ne, 5), 'float64')
        AUXG = N.empty((ng, 5), 'float64')
        print 'observations: SSG + GRAV'

        # a) signal: GRAV | covs: C_ll, C_mm, C_gg, C_gl
        if signal == 'g':
            COVLL = N.loadtxt(fcovll)
            COVMM = N.loadtxt(fcovmm)
            COVGG = N.loadtxt(fcovgg)
            COVGL = N.loadtxt(fcovgl)
            print 'signal: GRAV'
            print 'covariances: C_ll, C_mm, C_gg, C_gl'
        # b) signal: GEOID | covs: C_ll, C_mm, C_gg, C_gl, C_ng, C_nl
        elif signal == 'n':
            COVLL = N.loadtxt(fcovll)
            COVMM = N.loadtxt(fcovmm)
            COVGG = N.loadtxt(fcovgg)
            COVGL = N.loadtxt(fcovgl)
            COVNG = N.loadtxt(fcovng)
            COVNL = N.loadtxt(fcovnl)
            print 'signal: GEOID'
            print 'covariances: C_ll, C_mm, C_gg, C_gl, C_ng, C_nl'
        else:
            print 'Error with signal choice: -s[g|n]'
            exit()

    xmin, xmax, ymin, ymax = region

    # change longitude: -180/180 -> 0/360
    if xmin < 0:
        xmin += 360.
    if xmax < 0:
        xmax += 360.

    if scale == True:
        print 'using cell scale factor (local covariances)'
    else:
        s = 1.0

    if not d == None:
        print 'circular inversion cell: d =', d, '(deg)'
        print 'min points per inversion cell:', MIN_OBS_PER_CELL
    elif not l == None:
        print 'square inversion cell: l =', l, '(deg)'
        print 'min points per inversion cell:', MIN_OBS_PER_CELL
    else:
        print 'Error: diameter (-d) or side (-l) of cell is missing!'
        exit()

    # grid calculation ----------------------------------------------------

    TEMP = N.empty(4, 'float64')
    GRID = N.empty((0, 4), 'float64')

    print "grid spacing: %.2f' x %.2f'" % (dx*60.0, dy*60.0)
    print 'calculating grid: %.2f/%.2f/%.2f/%.2f ...' % region

    for lat in N.arange(ymin, ymax, dy):
        for lon in N.arange(xmin, xmax, dx):

            # select points inside the inversion cell

            if filegrav == None:
                if not d == None:
                    nobs = select_points(DATAE, AUXE, lon, lat, d)  # circular
                else:
                    nobs = select_points2(DATAE, AUXE, lon, lat, l) # square
            else:
                if not d == None:
                    ne = select_points(DATAE, AUXE, lon, lat, d)  
                    ng = select_points(DATAG, AUXG, lon, lat, d)
                else:
                    ne = select_points2(DATAE, AUXE, lon, lat, l)  
                    ng = select_points2(DATAG, AUXG, lon, lat, l)
                nobs = ne + ng

            # if there are sufficient observations inside the cell 
            # (i.e. it is a valid point of the grid outside the continent):
            # fills obs, C_sig (cov_sig_obs), C_OBS (cov_obs_obs)

            if nobs >= MIN_OBS_PER_CELL:  # num of sufficient obs for signal prediction
                #print nobs
                
                ### (1) obsservations: SSG
                if filegrav == None or ng == 0:        # no GRAV observations
                    OBS = AUXE[:nobs,:]                # SSGs
                    obs = OBS[:,2]                     # VECTOR of observations
                    C_sig = N.empty(nobs, 'float64')   # VECTOR of signal covs
                    C_OBS = N.empty((nobs,nobs), 'float64') # MATRIX of obs covs + err

                    # cell scale factor
                    if scale == True:
                        s = scale_factor(obs, COVLL, nobs)

                    # a) signal: GRAV
                    if signal == 'g':
                        fills_cov_signal1(OBS, COVGL, C_sig, lon, lat, s)
                        fills_cov_observ1(OBS, COVLL, COVMM, C_OBS, s)

                    # b) signal: GEOID
                    elif signal == 'n':
                        fills_cov_signal1(OBS, COVNL, C_sig, lon, lat, s)
                        fills_cov_observ1(OBS, COVLL, COVMM, C_OBS, s)

                ### (2) observations: SSG + GRAV
                else:					
                    OBS = AUXE[:ne,:]
                    OBS = N.vstack((OBS, AUXG[:ng,:]))
                    obs = OBS[:,2]                          
                    C_sig = N.empty(nobs, 'float64')        
                    C_OBS = N.empty((nobs,nobs), 'float64') 

                    # cell scale factor
                    if scale == True:
                        s = scale_factor(obs, COVLL, ne, COVGG, ng)

                    # a) signal: GRAV
                    if signal == 'g':
                        fills_cov_signal2(OBS, COVGL, COVGG, C_sig, lon, lat, \
                                          ne, ng, s)
                        fills_cov_observ2(OBS, COVLL, COVMM, COVGG, COVGL, C_OBS, \
                                          ne, ng, s)

                    # b) signal: GEOID
                    elif signal == 'n':
                        fills_cov_signal2(OBS, COVNL, COVNG, C_sig, lon, lat, \
                                          ne, ng, s)
                        fills_cov_observ2(OBS, COVLL, COVMM, COVGG, COVGL, C_OBS, \
                                          ne, ng, s)
    			 
                ### solves the LSC system: sig = C_sig * C_OBS_inv * obs
                ### for each point (cell) in the grid

                # (1) using the Hwang and Parsons (1995) algorithm
                try:
                    sig, err = LSC_solver1(C_sig, C_OBS, obs, var_sig)

                # (2) using conventional matrix inversion
                except:
                    sig, err = LSC_solver2(C_sig, C_OBS, obs, var_sig)

                # mark to indicate an anomalous value
                if N.abs(sig) > MAX_SIGNAL_VAL:
                    sig = ANOMALOUS_VAL
                    err = ANOMALOUS_VAL

            # if no obs or not sufficient -> signal = 0 -------------------
            else:
                sig = 0.0         
                err = 0.0

            TEMP[0] = lon                  # lon grid point
            TEMP[1] = lat                  # lat grid point
            TEMP[2] = sig                  # signal on grid point
            TEMP[3] = err                  # error on grid point
    
            GRID = N.vstack((GRID, TEMP))

    print 'saving data ...'
    N.savetxt(fileout, GRID, fmt='%f', delimiter=' ')
    print 'output [lon,lat,signal,error] -> ' + fileout


if __name__ == '__main__':
    main()
