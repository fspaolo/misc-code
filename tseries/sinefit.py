import sys
import numpy as np
import matplotlib.pyplot as plt
from pandas import *
from math import *
from patsy import dmatrix
from  sklearn.preprocessing import PolynomialFeatures as poly

import altimpy as ap


sin_data = DataFrame({'x' : np.linspace(0, 1, 101)})
noise = np.random.normal(0, 0.5, 101)
#noise = np.random.uniform(-1, 1, 101)
sin_data['y'] = np.sin(2 * pi * sin_data['x']) + noise
x = sin_data['x'].values
y = sin_data['y'].values
X = dmatrix('C(x, Poly)')

N = 5

w = 1/noise
out = ap.lstsq_cv(x, y, cv=10, max_deg=N, weight=w, randomise=True, return_coef=True)
y_wls, coef, deg, mse, var = out

y_ols = ap.lstsq_cv(x, y, cv=10, max_deg=N, weight=None, randomise=True)

a2 = np.polyfit(x, y, 1, w=None)#w)
y_line = np.polyval(a2, x)

m, c = ap.linear_fit(x, y, return_coef=True)
m2, c2 = ap.linear_fit_robust(x, y, return_coef=True)

out = ap.lasso_cv(x, y, cv=10, max_deg=N, return_model=True)
y_lasso, lasso = out

a = np.append(lasso.intercept_, lasso.coef_)

#y_lasso = np.dot(X[:,:N+1], a)
#dy_lasso = a[1] * X[:,0] + 2 * a[2] * X[:,1] # + 3 * a[3] * X[:,2]
dy_lasso = np.gradient(y_lasso, x[2] - x[1])

print a[1]
print 'coef.:', a
print 'slope:', y_lasso[-1] - y_lasso[0]

plt.figure()
plt.errorbar(x, y, yerr=noise, fmt='x')
'''
plt.plot(x, y_wls, label='wls')
plt.plot(x, y_ols, label='ols')
plt.plot(x, y_line, label='linear')
'''
plt.plot(x, y_lasso, label='lasso.')
plt.plot(x, dy_lasso, 'k', label='dy/dx.')
#plt.plot(x, dy_lasso2, 'k--', label='dy/dx.')
plt.legend()
plt.show()
