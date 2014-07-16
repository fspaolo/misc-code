#!/urs/bin/env python

#    covfit.py - Program to (1) prepare (automaticaly) de input file 
#    for covfit8.f, (2) format de output file in adequate columns for 
#    using the results (in the LSC and plotting) and (3) transform
#    the units of modeled covs from arcseconds to microradians.
#    
#    Author:	Fernando Paolo
#    Date:		Feb/2009
#    Usage:		python covfit.py -h

# editable part --------------------------------------------------------
head = """\
 F
 4 f f t f
 2
 4
 -6.0 230.0 360 f t f
 0 1 0.2 f
edgv.osu89
 8 1.0e0 1.0e0 1.0e0
 1
 %s 3 1 0.0 0.0 1 1.0 F
230.0 230.0 -53.0 -29.0 0.01 -7.0 7.0 0.01
"""
#-----------------------------------------------------------------------
# Note:
# code: 1=heigh anomaly, 3=gravity anomaly, 6=dov-xi, 7=dov-eta
# OBS: the step size changes the covariance function a lot!!!

import numpy as N
import optparse as op
import subprocess as sp
import shutil as sh
from sys import exit

usage = "python %prog <filein> [options]"
parser = op.OptionParser(usage=usage)
parser.add_option('-c', 
    	          dest='cols',
                  default='(0,1,2)', 
                  help='columns to be loaded [dist,cov,nprod]: -c0,1,2',
                  )
parser.add_option('-o', 
    	          dest='fileout',
                  default='covfit.out', 
                  help='writes the output to FILEOUT [dist,cov]: -ocovfit.out',
                  )
parser.add_option('-r', 
				  dest="urad", 
                  action="store_true", 
				  default=False,
                  help='converts arcsecond to microradian: -r',
                  )
parser.add_option('-R', 
				  dest="urad2", 
                  action="store_true", 
				  default=False,
                  help='converts arcsecond**2 to microradian**2: -R',
                  )
parser.add_option('-p', 
				  dest="plot", 
                  action="store_true", 
				  default=False,
                  help='plots the result: -p',
                  )
options, remainder = parser.parse_args()

if len(remainder) < 1:
	print 'The input file is missing!\n'
	parser.print_help()
	exit()

filein = remainder[0]       # file with covariances
cols = eval(options.cols)   # columns to be loaded
fileout = options.fileout   # output file
covfit_in = 'covfit8.in'    # formated input file for covfit8.f
covfit_out = 'covfit8.out'  # covfit8.f's output file
plot = options.plot         # for ploting the resuls
urad = options.urad         # for converting arcsec to urad
urad2 = options.urad2       # for converting arcsec**2 to urad**2

#-----------------------------------------------------------------------

r = 4.8481368169032191   # 1 arcsec = 4.8481368169032191 urad
R = 23.504430595412476   # 1 arcsec**2 = 23.504430595412476 urad**2

covs = N.loadtxt(filein, usecols=cols)
ncov = covs.shape[0]

# set the number of covariances (data) in the head
head = head % ncov


def format_input(covs, covfit_in, head, ncov):
    covin = open(covfit_in, 'w')
    covin.write(head)
    for i in N.arange(ncov):
        x = covs[i,0]
        y = covs[i,1]
        z = int(covs[i,2])
        covin.write('%s %s %s\n' % (x,y,z))
    covin.close()


def format_output(covfit_out):
    covout = open(covfit_out, 'r')
    line = covout.readline()
    while line:                            # while there is a line to be read
        line = covout.readline()
        if line.find('PSI') != -1:         # finds a line with the word 'PSI'
            covout.readline()              # jumps a blank line
            fout = open(fileout, 'w')      # always erase last record
            for i in N.arange(ncov):       # scan all modeled covs
                line = covout.readline()
                lin = line.split()
                dist = lin[3]
                tmp = lin[7]               # to prevent error numbers
                dot = tmp.find('.')        # finds the first dot                
                cov = tmp[:dot+5]          # copy the number with only 4 decimals

                dist = float(dist)         # str to float
                cov = float(cov)           # str to float
                # converts arcsec to urad
                if urad == True:              
                    fout.write('%f %f\n' % (dist, cov * r))
                # converts arcsec**2 to urad**2
                elif urad2 == True:            
                    fout.write('%f %f\n' % (dist, cov * R))
                # writes default units
                else:                      
                    fout.write('%f %f\n' % (dist, cov))
            fout.close()
    covout.close()


def plot_covs(filein, fileout):
    import pylab as P
    data1 = P.load(filein)
    data2 = P.load(fileout)
    P.plot(data1[:,0], data1[:,1], 'o')
    P.plot(data2[:,0], data2[:,1])
    P.grid('True')
    P.show()


def main():
    format_input(covs, covfit_in, head, ncov)
    sp.call('./covfit8 < %s > %s' % (covfit_in, covfit_out), shell=True)
    format_output(covfit_out)
    finfo = fileout[:-3] + 'info'    # changes the extension
    sh.move(covfit_out, finfo)       # moves to fileout dir
    print 'Output [dist,cov] -> ' + fileout
    print 'Output (info)     -> ' + finfo

    if plot == True:
        plot_covs(filein, fileout)


if __name__ == '__main__':
    main()
