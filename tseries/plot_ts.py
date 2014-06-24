import sys
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as img
import matplotlib.dates as mdates

sys.path.append('/Users/fpaolo/code/misc')
import util
import viz

# input
#---------------------------------------------------------------------
#files = sys.argv[1:]

TITLE = ''
SAVEFIG = 'ross3'
LABEL = 'Elevation change rate (m/yr)'

y1, h1 = np.loadtxt('ross1_ts.txt', unpack=True)
y2, h2 = np.loadtxt('ross2_ts.txt', unpack=True)
y3, h3 = np.loadtxt('ross3_ts.txt', unpack=True)
y4, h4 = np.loadtxt('ross4_ts.txt', unpack=True)
y5, h5 = np.loadtxt('ross5_ts.txt', unpack=True)
y6, h6 = np.loadtxt('ross6_ts.txt', unpack=True)


def fy2ymd(fyear):
    """Decimal year -> year, month."""
    fy, y  = np.modf(fyear)
    fmonth, y = fy*12, int(y)
    fm, m  = np.modf(fmonth)
    d, m = int(np.ceil(fm*30.4375)), int(np.ceil(fm))
    return [y, m, d]

def fy2dt(fyears):
    ymd = np.asarray([fy2ymd(y) for y in fyears])
    return np.asarray([dt.date(y, m, d) for y, m, d in ymd])

def linear_fit(x, y, return_coef=False):
    """
    Fit a straight-line by Ordinary Least Squares.

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    x, y = np.asarray(x), np.asarray(y)
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    if len(ind) < 2:
        return (np.nan, np.nan)
    x, y = x[ind], y[ind]
    A = np.ones((len(x), 2))
    A[:,0] = x
    m, c = np.linalg.lstsq(A, y)[0]
    if return_coef:
        return (m, c)
    else:
        x_pol = np.linspace(x.min(), x.max(), 200)
        y_pol = m*x_pol + c
        return (x_pol, y_pol)


def plot_ts(ax, x, y, dtime, title='', savefig=None, linewidth=None, color=None, alpha=None, label=True):
    #x_pol, y_pol = util.poly_fit(x, y, order=3)
    x_lin, y_lin = linear_fit(x, y)
    m, c = linear_fit(x, y, return_coef=True)
    
    x = x.round(2)
    #n = y_lin[0]  # zero at the begining 
    n = y_lin[0] + (y_lin[-1] - y_lin[0])/2.  # zero at the middle 
    x_dt = fy2dt(x)
    ax.plot(x, y-n, linewidth=linewidth, color=color, alpha=alpha)
    #plt.plot(x_pol, y_pol-n, **kw)
    x_lin_dt = np.sort(fy2dt(x_lin))
    ax.plot(x_lin, y_lin-n, linewidth=linewidth-1, color=color, alpha=alpha)

    if label:
        viz.add_inner_title(ax, 'dh/dt = %.1f cm/yr' % (m*100), loc=1)
        viz.add_inner_title(ax, title, loc=2)
    plt.ylim(-.28, .28)
    plt.xlim(2003.5, 2010.1)
    #plt.xlabel('years')
    plt.ylabel('Elevation change (m)')
    if savefig is not None:
        pass
        #plt.savefig(savefig+'_ts.png')
        #np.savetxt(savefig+'_ts.txt', np.column_stack((x, y)))
    return ax

#---------------------------------------------------------------------

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot((111))
'''
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
'''
t, ts, y = fy2dt(y1), h1, y1
ax = plot_ts(ax, y, ts, t, title=TITLE, savefig=SAVEFIG, linewidth=3, color='k', alpha=0.3, label=False)
t, ts, y = fy2dt(y2), h2, y2
ax = plot_ts(ax, y, ts, t, title=TITLE, savefig=SAVEFIG, linewidth=3, color='k', alpha=0.3, label=False)
t, ts, y = fy2dt(y3), h3, y3
ax = plot_ts(ax, y, ts, t, title=TITLE, savefig=SAVEFIG, linewidth=3, color='k', alpha=0.3, label=False)
t, ts, y = fy2dt(y4), h4, y4
ax = plot_ts(ax, y, ts, t, title=TITLE, savefig=SAVEFIG, linewidth=3, color='k', alpha=0.3, label=False)
t, ts, y = fy2dt(y5), h5, y5
ax = plot_ts(ax, y, ts, t, title=TITLE, savefig=SAVEFIG, linewidth=3, color='k', alpha=0.3, label=False)
t, ts, y = fy2dt(y6), h6, y6
ax = plot_ts(ax, y, ts, t, title=TITLE, savefig=SAVEFIG, linewidth=3, color='b', alpha=1.0, label=True)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#ax.set_xticks(t) 
#fig.autofmt_xdate()

plt.savefig('ross6_ts.png')
plt.show()
