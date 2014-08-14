### begin colormap_slider.py ################################# 
import math, copy 
import numpy 
from matplotlib import pyplot, colors, cm 
from matplotlib.widgets import Slider 

def cmap_powerlaw_adjust(cmap, a): 
    ''' 
    returns a new colormap based on the one given 
    but adjusted via power-law: 

    newcmap = oldcmap**a 
    ''' 
    if a < 0.: 
        return cmap 
    cdict = copy.copy(cmap._segmentdata) 
    fn = lambda x : (x[0]**a, x[1], x[2]) 
    for key in ('red','green','blue'): 
        cdict[key] = map(fn, cdict[key]) 
        cdict[key].sort() 
        assert (cdict[key][0]<0 or cdict[key][-1]>1), \
            "Resulting indices extend out of the [0, 1] segment." 
    return colors.LinearSegmentedColormap('colormap',cdict,1024) 

def cmap_center_adjust(cmap, center_ratio): 
    ''' 
    returns a new colormap based on the one given 
    but adjusted so that the old center point higher 
    (>0.5) or lower (<0.5) 
    ''' 
    if not (0. < center_ratio) & (center_ratio < 1.): 
        return cmap 
    a = math.log(center_ratio) / math.log(0.5) 
    return cmap_powerlaw_adjust(cmap, a) 

def cmap_center_point_adjust(cmap, range, center): 
    ''' 
    converts center to a ratio between 0 and 1 of the 
    range given and calls cmap_center_adjust(). returns 
    a new adjusted colormap accordingly 
    ''' 
    if not ((range[0] < center) and (center < range[1])): 
        return cmap 
    return cmap_center_adjust(cmap, 
        abs(center - range[0]) / abs(range[1] - range[0])) 


if __name__ == '__main__': 
    ### create some 2D histogram-type data 
    def func3(x,y): 
        return (1- x/2 + x**5 + y**3)*numpy.exp(-x**2-y**2) 
    x = numpy.linspace(-3.0, 3.0, 60) 
    y = numpy.linspace(-3.0, 3.0, 60) 
    X,Y = numpy.meshgrid(x, y) 
    Z = func3(X, Y) 
    extent = [x[0],x[-1],y[0],y[-1]] 


    plotkwargs = { 
        'extent' : extent, 
        'origin' : 'lower', 
        'interpolation' : 'nearest', 
        'aspect' : 'auto'} 

    ### interactively adjustable with a slider 
    fig = pyplot.figure(figsize=(6,4)) 
    fig.subplots_adjust(top=0.8) 
    ax = fig.add_subplot(1,1,1) 
    cmap = cm.seismic 
    plt = ax.imshow(Z, cmap=cmap, **plotkwargs) 
    cb = fig.colorbar(plt, ax=ax) 

    axcmap = fig.add_axes([0.1, 0.85, 0.8, 0.05], axisbg='white') 
    scmap = Slider(axcmap, '', 0.0, 1.0, valinit=0.5) 

    def update(val): 
        cmapcenter = scmap.val 
        #plt.set_cmap(cmap_powerlaw_adjust(cmap, 2)) 
        #plt.set_cmap(cmap_center_adjust(cmap, cmapcenter)) 
        plt.set_cmap(cmap_center_point_adjust(cmap, Z.ravel(), -0.5)) 
    scmap.on_changed(update) 



    pyplot.show() 
### end colormap_slider.py ###################################
