#!/bin/env python
"""
Matplotlib module enables hillshade method (shade) using a LightSource class (v 0.99).
The problem is it uses the data itself as intensity and data. It is very useful for viewing a DEM but sometimes you would like the DEM as intensity underlying some other data. Another problem is that the shade method is producing a very light colored image sometimes even white where intensity is high.
I used as an example a DEM derived from SRTM v4 data acquired at the International  Centre for Tropical  Agriculture (CIAT - http://srtm.csi.cgiar.org) the hillshade production was made using LightSource class with azimuth of 165 deg. and altitude of 45 deg.)

The difference in the shading colors derived from the method used to produce it. While the matplotlib method uses "hard light" method I use a "soft light" method. the matplotlib is converting the RGB colors to HSV and then calculate the new saturation and value according to the intensity. I use a formula based on the description of ImageMagick's pegtop_light.which is much faster as it is a single formula. Another advantage is the option to use a separate layer as the intensity and another as the data used for colors.

Modified from:
http://rnovitsky.blogspot.com/2010/04/using-hillshade-image-as-intensity.html

"""

from pylab import *

def shade(a, intensity=None, cmap=cm.jet, scale=10.0, azdeg=165.0, altdeg=45.0):
    '''sets shading for data array based on intensity layer 
      or the data's value itself.
    inputs:
      a - a 2-d array or masked array
      intensity - a 2-d array of same size as a (no chack on that)
                        representing the intensity layer. if none is given
                        the data itself is used after getting the hillshade values
                        see hillshade for more details.
      cmap - a colormap (e.g matplotlib.colors.LinearSegmentedColormap
                  instance)
      scale,azdeg,altdeg - parameters for hilshade function see there for
                  more details
    output:
      rgb - an rgb set of the Pegtop soft light composition of the data and 
               intensity can be used as input for imshow()
    based on ImageMagick's Pegtop_light:
    http://www.imagemagick.org/Usage/compose/#pegtoplight'''
    if intensity is None:
        # hilshading the data
        intensity = hillshade(a,scale=scale,azdeg=azdeg,altdeg=altdeg)
    else:
        # or normalize the intensity
        intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    # get rgb of normalized data based on cmap
    rgb = cmap((a-a.min())/float(a.max()-a.min()))[:,:,:3]
    # form an rgb eqvivalent of intensity
    d = intensity.repeat(3).reshape(rgb.shape)
    # simulate illumination based on pegtop algorithm.
    rgb = 2*d*rgb+(rgb**2)*(1-2*d)
    return rgb


def hillshade(data, scale=10.0, azdeg=165.0, altdeg=45.0):
    ''' convert data to hillshade based on matplotlib.colors.LightSource class.
    input: 
         data - a 2-d array of data
         scale - scaling value of the data. higher number = lower gradient 
         azdeg - where the light comes from: 0 south ; 90 east ; 180 north ;
                      270 west
         altdeg - where the light comes from: 0 horison ; 90 zenith
    output: a 2-d array of normalized hilshade 
    '''
    # convert alt, az to radians
    az = azdeg*pi/180.0
    alt = altdeg*pi/180.0
    # gradient in x and y directions
    dx, dy = gradient(data/float(scale))
    slope = 0.5*pi - arctan(hypot(dx, dy))
    aspect = arctan2(dx, dy)
    intensity = sin(alt)*sin(slope) + cos(alt)*cos(slope)*cos(-az - aspect - 0.5*pi)
    intensity = (intensity - intensity.min())/(intensity.max() - intensity.min())
    return intensity
