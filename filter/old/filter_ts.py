#!/usr/bin/env python
doc = """\
Filter a 1-D time series in time or frequency domain.
""" 
# Fernando Paolo <fpaolo@ucsd.edu>
# August 15, 2011

import numpy as np
import tables as tb
import pylab as pl
import argparse as ap
import scipy.signal as sg
import sys
import os

# parse command line arguments
parser = ap.ArgumentParser(description=doc)
group = parser.add_mutually_exclusive_group()
parser.add_argument('files', nargs='*', help='HDF5/ASCII file[s] to read')
parser.add_argument('-t', dest='timecol', type=int, default=0, 
    help='column of variable time (0,1..) [default: 0]')
parser.add_argument('-x', dest='varcol', type=int, default=1, 
    help='column of variable to filter (0,1..) [default: 0]')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='reads and writes ASCII files [default: HDF5]')

args = parser.parse_args()

class GetData(object):
    """Class to open a file and get the 2-D data (ASCII or HDF5).
    """
    def __init__(self, fname=None):
        if fname is None:
            print 'no file name given!'
            sys.exit()
        self.fname = fname
        self._getdata()
       
    def _getdata(self):
        try:
            # load ASCII
            self.data = np.loadtxt(self.fname)    
            print 'open ASCII file'
        except:
            # load HDF5
            h5f = tb.openFile(self.fname)         
            self.data = h5f.root.data.read()
            h5f.close()
            print 'open HDF5 file'


class PrepareData(object):
    """Class to prepare the time series for filtering/smoothing.
    """
    def __init__(self, data):
        self.data = data 

    def ym2fyear(self, ycol=0, mcol=1):
        """Converts year and month to fractional year.
        """
        self.fyear = self.data[:,ycol] + self.data[:,mcol]/12.

    def sort(self, col=None):
        """Sort data by `fyear` (if exists) or by column `col`.
        """
        if hasattr(self, 'fyear') and col is None:
            i = np.argsort(self.fyear)
            self.fyear = self.fyear[i]
            self.data = self.data[i,:]
            print 'using column `fyear` to sort data'
        else:
            # 0 is default if `col` not specified
            if col is None:    
                col = 0
            i = np.argsort(self.data[:,col])
            self.data = self.data[i,:]
            print 'using column `%d` to sort data' % col

    def average(self, col=None):
        """Average repeated values of given `col` in a rank-2 array.
        """
        if hasattr(self, 'fyear') and col is None:
            print 'using column `fyear` to average repeated values'
            for elem in self.fyear:
                # 1) find repeated values
                i, = np.where(self.fyear == elem)    
                if i.shape[0] > 1:
                    # 2) average repeated values
                    self.data[i,:] = np.average(self.data[i,:], axis=0)
            # 3) exclude repeated values
            self.fyear, ind = np.unique(self.fyear, return_index=True)
            self.data = self.data[ind,:] 
        else:
            if col is None:
                col = 0
            print 'using column `%s` to average repeated values' % col
            for elem in self.data[:,col]:
                i, = np.where(self.data[:,col] == elem)
                if i.shape[0] > 1:
                    self.data[i,:] = np.average(self.data[i,:], axis=0)
            _, ind = np.unique(self.data[:,col], return_index=True)
            self.data = self.data[ind,:]

    def interp(self, dt=1, col=None):
        """Fill in the gaps of a 1-D time series using linear interpolation.
        """
        # convert to fraction of a year
        dt = dt/12.                          
        if hasattr(self, 'fyear') and col is None:
            # check if data is increasing
            if not np.all(np.diff(self.fyear) > 0):  
                print 'time data must be increasing!'
                sys.exit()
            # points to interpolate
            t1, t2 = self.fyear[0], self.fyear[-1]
            time = np.arange(t1, t2+dt, dt)          
            dataout = np.empty((time.shape[0], self.data.shape[1]), 'f8')
            # linear interpolation of each column in `data`
            for j in range(self.data.shape[1]):
                dataout[:,j] = np.interp(time, self.fyear, self.data[:,j])
            self.fyear = time
            self.data = dataout

    def hist(self, bins=100, col=4):
        """Plot histogram of detrended data.
        """
        from pylab import hist, show
        x = sg.detrend(self.data[:,col])
        hist(x, bins=bins)
        show()

    def cutoff(self, val=9999, col=4):
       """Cut off values larger than `abs(val)`.
       """
       x = sg.detrend(self.data[:,col])
       i, = np.where(np.abs(x) <= np.abs(val))
       self.data = self.data[i,:]
       if hasattr(self, 'fyear'):
           self.fyear = self.fyear[i]


class FilterData(object):
    """Performs filtering and smoothing of `columns` of 2-D data set.
    """
    def __init__(self, data):
        self.data = data 

    #-------------------------------------------------------------------

    def time_filt(self, x, window_len=13, window_type='hanning'):
        """
        smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window 
        with the signal. The signal is prepared by introducing reflected 
        copies of the signal (with the window size) in both ends so that 
        transient parts are minimized in the beginning and end part of 
        the output signal.
        
        From: http://www.scipy.org/Cookbook/SignalSmooth
        
        Parameters
        ----------
        x : the input signal.
        window_len : the dimension of the smoothing window.
            Should be an odd integer, e.g. period to filter plus one (T + 1).
        window : the type of window from `flat`, `hanning`, `hamming`, 
            `bartlett`, `blackman`. flat window will produce a moving 
            average smoothing.
     
        Output
        ------
        y : the smoothed signal.
            
        Example
        -------
        >>> t = linspace(-2,2,0.1)
        >>> x = sin(t)+randn(len(t))*0.1
        >>> y = smooth(x)
        
        See also
        --------
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, 
        numpy.convolve

        scipy.signal.lfilter
     
        TODO 
        ----
        The window parameter could be the window itself if an array 
        instead of a string.
        """
        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, 'Window is one of `flat`, `hanning`, `hamming`, \
                `bartlett`, `blackman`'
        
        s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
        print 'size of original data:', x.shape[0]
        print 'size of `convolution-prepared` data:', s.shape[0]

        if window_type == 'flat':    
            # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.'+window_type+'(window_len)')
        
        y = np.convolve(w/w.sum(), s, mode='valid')

        # to discard added points at the ends
        r = (window_len-1)/2

        return y[r:-r]

    #-------------------------------------------------------------------

    def spectral_filt(self, t, x, cutoff_freq=1., stop_band=1., atten_db=60.):
        """Low-pass filter a time series.

        Design and use a low-pass FIR filter using functions from 
        scipy.signal. It applies a forward-backward filter (for zero 
        phase-shift) with replication of the signal at the ends (to 
        avoid `losses` in the filtered signal).
        
        From: http://www.scipy.org/Cookbook/FIRFilter

        Parameters
        ----------
        t : 1-D array
            The time/space domain of the series.
        x : 1-D array
            The signal.
        cutoff_freq : float
            The frequency limit for low-pass filtering.
        stop_band : float
            The `width` (in freq) of the stop band (around `cutoff_freq`).
        atten_db : float (in dB)
            The `slope` of the transition (from 1 to 0) in the `stop_band`.

        Output
        ------
        filtered_x : 1-D array
            The filtered inputed signal.
        """
        from numpy import cos, sin, pi, absolute, arange
        from scipy.signal import kaiserord, lfilter, firwin

        self.cutoff_freq = cutoff_freq
        self.stop_band = stop_band
        self.atten_db = atten_db

        # sampling rate (freq)
        self.sample_freq = 1. / (t[1]-t[0])
        print 'sample freq: %.3f samples per unit time' % self.sample_freq

        # The Nyquist rate of the signal
        self.nyq_freq = self.sample_freq / 2.0
        print 'Nyquist freq: %.3f' % self.nyq_freq

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.
        width = self.stop_band / self.nyq_freq
        print 'stop band width %.3f' % self.stop_band

        # The desired attenuation in the stop band, in dB (between 20-120).
        self.atten_db = atten_db
        print 'stop band attenuation %.3f dB' % self.atten_db

        # Compute the order and Kaiser parameter for the FIR filter.
        self.N, self.beta = kaiserord(atten_db, width)

        # The cutoff frequency of the filter.
        self.cutoff_freq = cutoff_freq
        print 'cutoff freq: %.3f' % self.cutoff_freq

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        self.b = firwin(self.N, self.cutoff_freq/self.nyq_freq, window=('kaiser', self.beta))
        self.a = np.array([1.0])

        # filter x with a `zero phase-shift` FIR filter.
        filtered_x = sg.filtfilt(self.b, self.a, x)

        return filtered_x

    #-------------------------------------------------------------------

    def plot_filter(self):
        """Plot the filter response.
        """
        from numpy import pi, absolute
        from scipy.signal import freqz
        import pylab as pl

        b, N, nyq_freq = self.b, self.N, self.nyq_freq

        #------------------------------------------------
        # Plot the FIR filter coefficients.
        #------------------------------------------------

        pl.figure(1)
        pl.plot(b, 'bo-', linewidth=2)
        pl.title('Filter Coefficients `b` (%d taps)' % N)
        pl.grid(True)

        #------------------------------------------------
        # Plot the magnitude response of the filter.
        #------------------------------------------------

        pl.figure(2)
        pl.clf()
        w, h = freqz(b, worN=8000)
        f = (w/pi)*nyq_freq
        pl.plot(f, absolute(h), linewidth=2)
        pl.xlabel('Frequency (1/time)')
        pl.ylabel('Gain')
        pl.title('Frequency Response')
        pl.ylim(-0.05, 1.05)
        pl.grid(True)
        
        # Upper inset plot.
        ax1 = pl.axes([0.42, 0.6, .45, .25])
        pl.plot(f, absolute(h), linewidth=2)
        pl.xlim(f.min(), f.max()/6)
        pl.ylim(0.9985, 1.0015)
        pl.grid(True)
        
        # Lower inset plot
        ax2 = pl.axes([0.42, 0.25, .45, .25])
        pl.plot(f, absolute(h), linewidth=2)
        pl.xlim(f.max()/6, f.max()/3)
        pl.ylim(0.0, 0.0025)
        pl.grid(True)

        pl.show()
       

plot = 1

def main():
    d = GetData(args.files[0])

    p = PrepareData(d.data)
    p.ym2fyear(ycol=0, mcol=1)
    p.sort()
    t0 = p.fyear[:]
    y0 = p.data[:,4]
    #p.hist()    
    p.cutoff(val=5)
    p.average()
    p.interp(dt=1)

    f = FilterData(p.data)
    #yf = f.time_filt(p.data[:,4], window_len=19, window_type='blackman')
    yf = f.spectral_filt(p.fyear, p.data[:,4],
                         cutoff_freq=1, stop_band=.7, atten_db=60.)
    f.plot_filter()

    t = p.fyear
    y = p.data[:,4]

    #y, t = sg.resample(y, 100, t)
    #yf = sg.medfilt(y, kernel_size=5)
    #yf = sg.wiener(y, mysize=13)
    #yy = sg.cspline1d(y)
    #yf = sg.cspline1d_eval(yy, t, dx=t[1]-t[0], x0=t[0])
    #a, b = sg.butter(10, .01)
    #yf = sg.lfilter(b, a, y)
        
    if plot:
        pl.errorbar(t0, y0, yerr=1.5, alpha=.5)
        pl.plot(t, yf, 'k', linewidth=4)
        #pl.plot(t0, y0, 'x', t, yf, 'k', linewidth=4)
        pl.savefig('ts.png')
        pl.show()

    #np.savetxt('test_ts.txt', np.column_stack((t, y, yf)), fmt='%.3f')
    
if __name__ == '__main__':
    main()
