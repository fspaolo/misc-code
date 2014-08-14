import brewer2mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


plt.rcParams['figure.figsize'] = (10, 2.5)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 8
plt.rcParams['patch.edgecolor'] = (0,0,0,0) 
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['patch.edgecolor'] = 'white'


names = [
    'Queen Maud',
    'Amery',
    'Wilkes-Victoria',
    'Ross',
    'Amundsen',
    'Bellingshausen',
    'Larsen',
    'Filchner-Ronne',
    'East Antarctica',
    'West Antarctica',
    'All Antarctica',
    ]

# doad data
FILE_RATES = '/Users/fpaolo/data/shelves/h_integrate.csv'
data = pd.read_csv(FILE_RATES, index_col=0)
data = data.ix[names]

#red = (0.78, 0.22, 0.18) # original
red = (199/255., 56/255., 46/255.) # RGB triplet
blue = (5/255., 113/255., 176/255.)
green = (1/255., 133/255., 113/255.)

color = [red if v < 0 else blue for v in data['dhdt_poly(cm/yr)']]

# area
#----------------------------------

fig = plt.figure(frameon=True, figsize=(10, 2.5))
fig.patch.set_facecolor('white')

var = data['total_area(km2)'].astype('int')
var2 = data['survey_area(km2)'].astype('float')
name = var.index
pos = np.arange(len(var))

var2 /= var / 100
var2 = var2.round(0).astype('int')

plt.subplot(131)

plt.title('Area (km^2)')
plt.barh(pos, var, color='0.4')

#add the numbers to the side of each bar
for v, p in zip(var, pos):
    plt.annotate(str(v) , xy=(v + 8000, p + .3), va='center')

#cutomize ticks
ticks = plt.yticks(pos + .5, name, va='bottom')
xt = plt.xticks()[0]
plt.xticks(xt, [' '] * len(xt))

#minimize chartjunk
remove_border(left=False, bottom=False)
plt.grid(axis = 'x', color ='white', linestyle='-')

#set plot limits
plt.ylim(pos.max() + 1, pos.min() - 1)
plt.xlim(0, var.max() + 10000)

# height
#----------------------------------

var = data['dhdt_poly(cm/yr)'].apply(np.abs).apply(np.round, decimals=1)
name = var.index
pos = np.arange(len(var))

ind, = np.where(var == 0)

plt.subplot(132)

plt.title('Height rate (cm/year)')
plt.barh(pos, var, color=color)

#add the numbers to the side of each bar
for v, p in zip(var, pos):
    plt.annotate(str(v), xy=(v + .7, p + .3), va='center')

#cutomize ticks
ticks = plt.yticks(pos + .5, name, va='bottom')
xt = plt.xticks()[0]
plt.xticks(xt, [' '] * len(xt))

#minimize chartjunk
remove_border(left=False, bottom=False)
plt.grid(axis = 'x', color ='white', linestyle='-')

#set plot limits
plt.ylim(pos.max() + 1, pos.min() - 1)
plt.xlim(0, var.max() + .8)

# mass
#----------------------------------

var = data['dmdt_poly(Gt/yr)'].apply(np.abs).apply(np.round, decimals=1)
name = var.index
pos = np.arange(len(data))

var[ind] = 0

plt.subplot(133)

# plot bars
plt.title('Mass rate (Gt/year)')
plt.barh(pos, var, color=color)

# add the numbers to the side of each bar
for v, p in zip(var, pos):
    plt.annotate(str(v), xy=(v + 1, p + .3), va='center')

# cutomize ticks
ticks = plt.yticks(pos + .5, name, va='bottom')
xt = plt.xticks()[0]
plt.xticks(xt, [' '] * len(xt))

# minimize chartjunk
remove_border(left=False, bottom=False)
plt.grid(axis = 'x', color ='white', linestyle='-')

# set plot limits
plt.ylim(pos.max() + 1, pos.min() - 1)
plt.xlim(0, var.max() + 1)

#-------------------

fig.subplots_adjust(left=0.12, right=0.95, bottom=0.02, top=0.95, wspace=0.75, hspace=0.5)
plt.savefig('barchart.png', dpi=150)
plt.show()
