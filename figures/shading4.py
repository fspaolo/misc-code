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

#Show effect of altitude
ls45 = LightSource(azdeg=0,altdeg=45)
rgb45 = ls45.shade(Z,plt.cm.gist_heat)
rgbnoise45 = ls45.shade(Znoise,plt.cm.gist_heat)
rgbclimits45 = ls45.shade(Znoise,plt.cm.gist_heat,vmin=oldmin,vmax=oldmax,
                      limit_elevation = False)
rgbellimits45 = ls45.shade(Znoise,plt.cm.gist_heat,vmin=oldmin,vmax=oldmax,
                      limit_elevation = True)
plt.figure(4,figsize=(12,10))
plt.subplot(221)
plt.imshow(rgb45)
plt.title('Original')
plt.xticks([]); plt.yticks([])
plt.subplot(222)
plt.imshow(rgbnoise45)
plt.title('With high and low spikes\nin upper left and lower right')
plt.xticks([]); plt.yticks([])
plt.subplot(223)
plt.imshow(rgbclimits45)
plt.title('With spikes\nColormap limited, elevation unchanged')
plt.xticks([]); plt.yticks([])
plt.subplot(224)
plt.imshow(rgbellimits45)
plt.title('With spikes\nColormap and elevation limited')
plt.xticks([]); plt.yticks([])

plt.show()
