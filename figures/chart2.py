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

#colorbrewer2 Dark2 qualitative color table
dark2_colors = brewer2mpl.get_map('Dark2', 'Qualitative', 7).mpl_colors

plt.rcParams['patch.edgecolor'] = (0,0,0,0) 
plt.rcParams['figure.figsize'] = (10, 6.5)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.color_cycle'] = dark2_colors
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 8
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['patch.facecolor'] = dark2_colors[0]
#plt.rcParams['font.family'] = 'StixGeneral'


names = [
    'Brunt',
    'Riiser',
    'Fimbul',
    'Lazarev',
    'Baudouin',
    'Prince Harald',
    'Amery',
    'West',
    'Shackleton',
    'Totten',
    'Moscow',
    'Holmes',
    'Dibble',
    'Mertz',
    'Cook',
    'Rennick',
    'Mariner',
    'Drygalski',
    'Ross EAIS',
    'Ross WAIS',
    'Sulzberger',
    'Nickerson',
    'Getz',
    'Dotson',
    'Crosson',
    'Thwaites',
    'Pine Island',
    'Cosgrove',
    'Abbot',
    'Venable',
    'Stange',
    'Bach',
    'Wilkins',
    'George VI',
    'Larsen B',
    'Larsen C',
    'Larsen D',
    'Ronne',
    'Filchner',
    #
    'Queen Maud',
    'Wilkes-Victoria',
    'Ross',
    'Amundsen',
    'Bellingshausen',
    'Larsen',
    'Filchner-Ronne',
    #
    'East Antarctica',
    'West Antarctica',
    'All Antarctica',
    ]

FILE_RATES = '/Users/fpaolo/data/shelves/h_integrate.csv'
data = pd.read_csv(FILE_RATES, index_col=0)
data = data.ix[names]

#red = (0.78, 0.22, 0.18) # original
red = (199/255., 56/255., 46/255.) # RGB triplet
blue = (5/255., 113/255., 176/255.)
green = (1/255., 133/255., 113/255.)

color = [red if v < 0 else blue for v in data['dhdt_poly(cm/yr)']]

# density
#----------------------------------

fig = plt.figure(frameon=True)
fig.patch.set_facecolor('white')

var = data['density(kg/m3)'].round(1)
name = var.index
pos = np.arange(len(var))

plt.subplot(131)

plt.title('Density (kg/m^3)')
plt.barh(pos, var, color='0.4')

#add the numbers to the side of each bar
for v, p in zip(var, pos):
    plt.annotate(str(v), xy=(v + 50, p + .3), va='center')

#cutomize ticks
ticks = plt.yticks(pos + .4, name, va='bottom')
xt = plt.xticks()[0]
plt.xticks(xt, [' '] * len(xt))

#minimize chartjunk
remove_border(left=False, bottom=False)
plt.grid(axis = 'x', color ='white', linestyle='-')

#set plot limits
plt.ylim(pos.max() + 1, pos.min() - 1)
plt.xlim(0, var.max() + 900)

# thickness
#----------------------------------

var = np.abs(data['thickness(m)']).round(1)
name = var.index
pos = np.arange(len(var))

plt.subplot(132)

plt.title('Thickness (m)')
plt.barh(pos, var, color='0.4')

#add the numbers to the side of each bar
for v, p in zip(var, pos):
    plt.annotate(str(v), xy=(v + 20, p + .3), va='center')

#cutomize ticks
ticks = plt.yticks(pos + .3, name, va='bottom')
xt = plt.xticks()[0]
plt.xticks(xt, [' '] * len(xt))

#minimize chartjunk
remove_border(left=False, bottom=False)
plt.grid(axis = 'x', color ='white', linestyle='-')

#set plot limits
plt.ylim(pos.max() + 1, pos.min() - 1)
plt.xlim(0, var.max() + 50)

# ratio
#----------------------------------

var = np.abs(data['H/Z']).round(2)
name = var.index
pos = np.arange(len(data))

plt.subplot(133)

# plot bars
plt.title('Freeboard/Thickness')
plt.barh(pos, var, color='0.4')

# add the numbers to the side of each bar
for v, p in zip(var, pos):
    plt.annotate(str(v), xy=(v + .01, p + .3), va='center')

# cutomize ticks
ticks = plt.yticks(pos + .5, name, va='bottom')
xt = plt.xticks()[0]
plt.xticks(xt, [' '] * len(xt))

# minimize chartjunk
remove_border(left=False, bottom=False)
plt.grid(axis = 'x', color ='white', linestyle='-')

# set plot limits
plt.ylim(pos.max() + 1, pos.min() - 1)
plt.xlim(0, var.max() + .08)

#-------------------

fig.subplots_adjust(left=0.12, right=0.95, bottom=0.02, top=0.95, wspace=0.75, hspace=0.5)
plt.savefig('barchart.png', dpi=150)
plt.show()
