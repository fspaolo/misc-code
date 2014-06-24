'''
Created on Mar 16, 2013

@author: tiago
'''

import numpy as np
import tables as tb
from scipy.interpolate import UnivariateSpline
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
import scipy.optimize as op
import matplotlib.pyplot as plt

import altimpy as ap

DIR = '/Users/fpaolo/data/shelves/' 
FNAME = 'all_19920716_20111015_shelf_tide_grids_mts.h5'

def ssqe(sm, s, npts):
	return np.sqrt(np.sum(np.power(s-sm,2)))/npts

def testGauss(x, y, s, npts):
	b = gaussian(39, 10)
	#ga = filtfilt(b/b.sum(), [1.0], y)
	ga = filters.convolve1d(y, b/b.sum())
	plt.plot(x, ga)
	print "gaerr", ssqe(ga, s, npts)
	return ga

def testButterworth(nyf, x, y, s, npts):
	b, a = butter(4, 1.5/nyf)
	fl = filtfilt(b, a, y)
	plt.plot(x,fl)
	print "flerr", ssqe(fl, s, npts)
	return fl

def testWiener(x, y, s, npts):
        sigma2 = None #np.var(y)
        print sigma2
	wi = wiener(y, mysize=29, noise=sigma2)
	plt.plot(x,wi,'g')
	print "wieerr", ssqe(wi, s, npts)
	return wi

def testSpline(x, y, s, npts):
	sp = UnivariateSpline(x, y, s=240)
	plt.plot(x,sp(x),'r')
	print "splerr", ssqe(sp(x), s, npts)
	return sp(x)

def plotPowerSpectrum(y, w):
	ft = np.fft.rfft(y)
	ps = np.real(ft*np.conj(ft))*np.square(dt)
	plt.plot(w, ps)

if __name__ == '__main__':
    i, j = 1, 161
    f = tb.openFile(DIR + FNAME)
    x = ap.num2year(f.root.time_xcal[:])
    y =  f.root.dh_mean_xcal_interp_short_const[:,i,j]
    sigma = np.var(y)
    s = y[:]

    npts = len(x)
    end = x[-1]
    dt = end/float(npts)
    nyf = 0.5/dt
    '''
    sigma = 0.5
    x = np.linspace(0,end,npts)
    r = np.random.normal(scale = sigma, size=(npts))
    s = np.sin(2*np.pi*x)#+np.sin(4*2*np.pi*x)
    y = s + r
    '''
    plt.plot(x,s)
    plt.plot(x,y,ls='none',marker='.')
    #ga = testGauss(x, y, s, npts)
    #fl = testButterworth(nyf, x, y, s, npts)
    wi = testWiener(x, y, s, npts)
    sp = testSpline(x, y, s, npts)
    #plt.legend(['true','meas','gauss','iir','wie','spl'], loc='upper center')
    plt.savefig("signalvsnoise.png")
    plt.show()
    plt.clf()
    w = np.fft.fftfreq(npts, d=dt)
    w = np.abs(w[:npts/2+1]) #only freqs for real fft
    plotPowerSpectrum(s, w)
    plotPowerSpectrum(y, w)
    plotPowerSpectrum(ga, w)
    plotPowerSpectrum(fl, w)
    plotPowerSpectrum(wi, w)
    plotPowerSpectrum(sp, w)
    plt.yscale('log')
    plt.xlim([0,10])
    plt.ylim([1E-8,None])
    plt.legend(['true','meas','gauss','iir','wie','spl'], loc='upper center')
    plt.savefig("spectra.png")
    plt.show()
