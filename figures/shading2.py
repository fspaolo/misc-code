import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from colors import *

# example showing how to make shaded relief plots
# like mathematica
# (http://reference.wolfram.com/mathematica/ref/ReliefPlot.html)
# or Generic Mapping Tools
# (http://gmt.soest.hawaii.edu/gmt/doc/gmt/html/GMT_Docs/node145.html)

# test data
X,Y=np.mgrid[-5:5:0.05,-5:5:0.05]
Z=np.sqrt(X**2+Y**2)+np.sin(X**2+Y**2)
# create light source object.
ls = LightSource(azdeg=0,altdeg=65)
# shade data, creating an rgb array.
rgb = ls.shade(Z,plt.cm.gist_heat)

#Show effect on noise
oldmin = Z.min()
oldmax = Z.max()
oldrange = oldmax - oldmin
Znoise = Z.copy()
Znoise[:10,:10] = oldmax + 10*oldrange
Znoise[-10:,-10:] = oldmin - 10*oldrange
rgbnoise = ls.shade(Znoise,plt.cm.gist_heat)

#Compare imshow and imshow with shading for noisy images
plt.figure(2,figsize=(12,6))
plt.subplot(121)
plt.imshow(Znoise,cmap=plt.cm.gist_heat)
plt.title('imshow\nHigh and low spikes\nin upper left and lower right')
plt.xticks([]); plt.yticks([])
plt.subplot(122)
plt.imshow(rgbnoise)
plt.title('imshow with shading\nHigh and low spikes\nin upper left and lower right')
plt.xticks([]); plt.yticks([])
rgbclimits = ls.shade(Znoise,plt.cm.gist_heat,vmin=oldmin,vmax=oldmax,
                      limit_elevation = False)
rgbellimits = ls.shade(Znoise,plt.cm.gist_heat,vmin=oldmin,vmax=oldmax,
                      limit_elevation = True)

plt.show()
