#!/usr/bin/python
# -*- coding: utf-8 -*-

r"""
=========================================================
Gaussian Processes regression: basic introductory example
=========================================================

A simple one-dimensional regression exercise computed in two different ways:

1. A noise-free case with a cubic correlation model
2. A noisy case with a squared Euclidean correlation model

In both cases, the model parameters are estimated using the maximum
likelihood principle.

The figures illustrate the interpolating property of the Gaussian Process
model as well as its probabilistic nature in the form of a pointwise 95%
confidence interval.

Note that the parameter ``nugget`` is applied as a Tikhonov regularization
of the assumed covariance between the training points.  In the special case
of the squared euclidean correlation model, nugget is mathematically equivalent
to a normalized variance:  That is

.. math::
   \mathrm{nugget}_i = \left[\frac{\sigma_i}{y_i}\right]^2

"""
print(__doc__)

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# Licence: BSD 3 clause

import sys
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl

from scipy.interpolate import griddata, interp2d, RectBivariateSpline, SmoothBivariateSpline

np.random.seed(1)


def f(X):
    """The function (field) to predict."""
    x, y = X[:,0], X[:,1]
    #return np.sin(np.sqrt(x**2 + y**2)) / (np.sqrt(x**2 + y**2))  # sombrero [-10,10]
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2  # [0,1]
    #return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y**2)**2

#----------------------------------------------------------------------
#  First the noiseless case (interpolation)
#----------------------------------------------------------------------
# num of observations
n_samples = 75

# num of prediction points on each axis: n_eval x n_eval
n_eval = 40

# range of x and y axis
a, b = 0, 1

# noiseless case -> "almost" no uncertainty in the observations
nugget = 10 * sys.float_info.epsilon

# bounds (assuming anisotropy)
theta0 = [10] * 2
thetaL = [1e-1] * 2
thetaU = [20] * 2

# Observation coordinates
pts1 = (b-a) * np.random.random(n_samples) + a
pts2 = (b-a) * np.random.random(n_samples) + a
X = np.column_stack((pts1, pts2))

# Observations
y = f(X)

# Prediction points to evaluate (2d mesh)
axis = np.linspace(a,b,n_eval)
xx, yy = np.meshgrid(axis, axis)
X_pred = np.column_stack((xx.ravel(), yy.ravel()))

# Instanciate a Gaussian Process model
gp = GaussianProcess(regr='constant', corr='squared_exponential', 
                     theta0=theta0, thetaL=thetaL, thetaU=thetaU, 
                     random_start=5, nugget=nugget, verbose=True)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x/y-axis (ask for MSE as well)
y_pred, MSE = gp.predict(X_pred, eval_MSE=True)
sigma = np.sqrt(MSE)

# Plot the function, the prediction and the 95% confidence interval 
# based on the MSE
pl.figure()
pl.imshow(y_pred.reshape(n_eval,n_eval), extent=(a,b,b,a), 
          interpolation='bicubic')
pl.scatter(X[:,0], X[:,1], c=y, s=40)
pl.title('Kriging: Noiseless data')

pl.figure()
pl.imshow(sigma.reshape(n_eval,n_eval), extent=(a,b,b,a), 
          interpolation='bicubic')

# spline
#----------------------------------------------------------------------
#y = y_pred.reshape(n_eval,n_eval)

'''
func = interp2d(X[:,0], X[:,1], y, kind='cubic')
zz = func(axis, axis)
'''

zz = griddata((X[:,0], X[:,1]), y, (axis[None,:], axis[:,None]), method='cubic') 
zz[np.isnan(zz)] = 0

rbs = RectBivariateSpline(axis, axis, zz, kx=5, ky=5)
z = rbs.ev(X[:,0], X[:,1])

'''
sbs = SmoothBivariateSpline(X_pred[:,0], X_pred[:,1], zz.ravel(), kx=5, ky=5)
z = sbs.ev(X[:,0], X[:,1])
'''

'''
pl.figure()
'''
pl.imshow(zz, extent=(a,b,b,a), interpolation='bicubic')
#pl.scatter(X_pred[:,0], X_pred[:,1], c=zz.ravel(), s=40)
pl.scatter(X[:,0], X[:,1], c=z, s=40, marker='o')
pl.title('Spline: Noiseless data')
pl.show()
sys.exit()

#----------------------------------------------------------------------
# now the noisy case (fitting)
#----------------------------------------------------------------------

# Noise, within 5% of the signal range
dy = 0.05 * (y.max() - y.min())
noise = np.random.normal(0, dy, y.shape)
y += noise

# Instanciate a Gaussian Process model
gp = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                     thetaL=1e-3, thetaU=1, verbose=True,
                     nugget=(dy / y)**2, # <<< main difference: the 'nugget'!
                     random_start=100)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(X_pred, eval_MSE=True)
sigma = np.sqrt(MSE)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
pl.figure()
pl.imshow(y_pred.reshape(n_eval,n_eval), extent=(a,b,b,a), 
          interpolation='bicubic')
pl.scatter(X[:,0], X[:,1], c=y, s=40)
pl.title('Kriging: Noisy data')

pl.figure()
pl.imshow(sigma.reshape(n_eval,n_eval), extent=(a,b,b,a), 
          interpolation='bicubic')

# spline
#----------------------------------------------------------------------

zz = griddata((X[:,0], X[:,1]), y, (axis[None,:], axis[:,None]), method='cubic') 

pl.figure()
pl.imshow(zz, extent=(a,b,b,a), interpolation='bicubic')
pl.scatter(X[:,0], X[:,1], c=y, s=40)
pl.title('Spline: Noisy data')

pl.show()
