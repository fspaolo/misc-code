import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altimpy as ap
 
def stepfilt(x, delta=3, window=7):
    assert window % 2 != 0, 'window size must be odd'
    n = window / 2
    v = np.r_[np.repeat(x[0], n), x, np.repeat(x[-1], n)] # expanded arr
    m = pd.rolling_median(v, window, center=True)         # filtered arr
    for i in range(len(m)-1):
        diff = m[i+1] - m[i]
        if np.abs(diff) > delta:
            v[i+1:] -= diff
    return v[n:-n], m[n:-n]
 

def stepfilt(x, delta=3, window=7):
    assert window % 2 != 0, 'window size must be odd'
    n = window / 2
    v = np.r_[np.repeat(x[0], n), x, np.repeat(x[-1], n)] # expand for w
    m = pd.rolling_median(v, window, center=True)         # median filter
    for i in range(len(m)-1):
        diff = m[i+1] - m[i]
        if np.abs(diff) > delta:
            v[i+1:] -= diff
    return v[n:-n], m[n:-n]
 

d = np.loadtxt('/Users/fpaolo/Desktop/peaksb0.csv', skiprows=1, delimiter=',')
#d = np.loadtxt('/Users/fpaolo/Desktop/junk0.csv', skiprows=1, delimiter=',')
x = d[:,0]
'''
x = np.array([0,0,0,5,0,0,0,0,0,-3,0,0,0,6,0,0,0,-3,5])
x[6:] -= 8
'''


'''
# steps
x[20:] += 5
x[60:] += -4
x[1] = -11
x[2] = -11

# peaks
x[50] = 5
x[40] -= 4
x[70] += 4
'''

x[np.isnan(x)] = 0
x2, m2 = stepfilt(x, delta=3, window=7)  # 3,7 (safest) or 4,5 (higher corr)
x2 = ap.hpfilt(x, 7)

plt.subplot(311)
plt.plot(x, 'b', label='original', linewidth=2)
plt.ylabel('original')
plt.grid(True)

plt.subplot(312)
plt.plot(x, 'g', label='shifted', linewidth=1)
plt.plot(m2, 'g', label='shifted', linewidth=3)
plt.ylabel('filtered')
plt.grid(True)

plt.subplot(313)
plt.plot(x, 'r', label='corrected', linewidth=1)
plt.plot(x2, 'r', label='corrected', linewidth=3)
plt.ylabel('corrected')
plt.grid(True)

plt.show()
