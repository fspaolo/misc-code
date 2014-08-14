import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import altimpy as ap

from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    """Shift the colormap to match the specified midpoint.
    
    A subclass of Normalize to define a custom normalization.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases here...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def remap_cmap(cmap, start=0, midpoint=0.5, stop=1, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave 
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax) 
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin)) 
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave 
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin)) 

    Credits
    -------
    Paul H (initial version)
    Horea Christian (additions/modifications)
    Fernando Paolo (additions/modifications)

    TODO
    ----
    Set 'start' and 'stop' dynamically.

    '''
    # find optimal values for 'start', 'midpoint' and 'stop' to match the data
    if np.ndim(midpoint) != 0:
        midpoint = np.asarray(midpoint)[~np.isnan(midpoint)]
        midpoint = abs(midpoint.min()) / float(abs(midpoint.max()) + \
                                               abs(midpoint.min())) 
    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False), 
        np.linspace(0.5, stop, 129, endpoint=True)
    ])
    # shifted index to match the midpoint of the data
    new_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    for ri, si in zip(reg_index, new_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict, 256)
    plt.register_cmap(cmap=newcmap)
    return newcmap


def test(cmap, pos1=0.1, pos2=.3, start=0, midpoint=0.5, stop=1, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave 
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax) 
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin)) 
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave 
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin)) 

    Credits
    -------
    Paul H (initial version)
    Horea Christian (additions/modifications)
    Fernando Paolo (additions/modifications)

    TODO
    ----
    Set 'start' and 'stop' dynamically.

    '''
    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False), 
        np.linspace(0.5, stop, 129, endpoint=True)
    ])

    pos1_index = np.hstack([
        np.linspace(start, pos1, 64, endpoint=False), 
        np.linspace(0.5, pos1, 65, endpoint=True)
    ])
    # shifted index to match the midpoint of the data
    pos2_index = np.hstack([
        np.linspace(0.0, pos2, 64, endpoint=False), 
        np.linspace(pos2, 1.0, 65, endpoint=True)
    ])

    reg_index = np.r_[pos1_index, reg_index[128:]]
    new_index = np.r_[pos2_index, reg_index[128:]]
    print reg_index
    print new_index

    exit()
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}
    for ri, si in zip(reg_index, new_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict, 256)
    plt.register_cmap(cmap=newcmap)
    return newcmap



"""
red = (199/255., 56/255., 46/255.) # RGB triplet
blue = (5/255., 113/255., 176/255.)


# Ice
#colors = [(0, 0, 1), (.5, .5, 1), (1, 1, 1)]
#colors = plt.cm.gray(np.linspace(0.2, 1, 128))
colors = [(240, 60, 43), (240, 60, 43), (255,255,255), (33,113,181), (33,113,181)]
position = [0, 0.2, 0.5, 0.8, 1]
cmap = ap.make_cmap(colors, position, n=21)
cmap = plt.cm.coolwarm_r


# test data 
X, Y = np.mgrid[-5:5:0.05, -5:5:0.05] 
z = X + Y + np.sin(X**2 + Y**2)
print z.min(), z.max()

rgb1 = ap.shade(z, cmap=cmap, azdeg=0, altdeg=45, scale=5)
rgb2 = ap.shade2(z, cmap=cmap, azdeg=0, altdeg=45, scale=5)

plt.subplot(121)
plt.imshow(rgb1, cmap=cmap, vmin=z.min(), vmax=z.max())
plt.colorbar(orientation='horizontal')
plt.subplot(122)
plt.imshow(rgb2, cmap=cmap, vmin=z.min(), vmax=z.max())
plt.colorbar(orientation='horizontal')
plt.show()
"""

'''
For a colormap dependent on the data *distribution* we give bounds as a
nonlinear monotonically increasing sequence. That is, we are not mapping the
colors by the value of the data, but by the index of the bounds (regions).
'''
data = np.array([[-2, 0], [1, 3]])

cmap = plt.cm.bwr                       # user defined cmap
bounds = [-2, -1, -.5, 0, .2, 1, 4]

midpoint = abs(bounds[0]) / float(abs(bounds[-1]) + abs(bounds[0])) 
'''
cmap = remap_cmap(cmap, midpoint=bounds)
cmap.set_over('.8')
cmap.set_under('.8')

norm = mc.BoundaryNorm(bounds, cmap.N)  # this is the key!

cmap = test(cmap, pos1=.1, pos2=.4)
'''

norm = MidpointNormalize(midpoint=0, vmin=-2, vmax=2)

plt.contourf(data, cmap=cmap, norm=norm, levels=np.linspace(-2, 4, 50))#, bounds=bounds, norm=norm, vmin=bounds[0], vmax=bounds[-1])
#plt.colorbar(ticks=bounds, spacing='proportional')#, drawedges=True, extend='both')
plt.colorbar(ticks=bounds)
plt.show()

