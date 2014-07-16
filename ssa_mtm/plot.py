import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altimpy as ap

ap.rcparams()

fname = '/Users/fpaolo/data/shelves/amundsen/amundsen_seasonal_mat.txt'
#fname = 'gisst.dat'


# plot PCs
'''
d = pd.read_table(fname+'_spc.tmp', header=None, sep=' ', skipinitialspace=True)
title = '%d Leading PCs from original data' % len(d.columns)
d.plot(subplots=True, linewidth=2, legend=False, title=title, yticks=[])

### 1) plot variance
d = pd.read_table(fname+'_spc.tmp_mssa_eigvar.tmp', header=None, sep=' ', 
                  skipinitialspace=True,
                  names=['rank', '%-var']
                  )
plt.figure()
plt.title('Variance per Mode')
d.plot(x='rank', y='%-var', logy=True, marker='o')
plt.ylabel('MSSA % Variance')
plt.xlabel('Rank K')
'''

### 2) plot singular spectrum (eigenval^1/2)
d = pd.read_table(fname+'_spc.tmp_mcmssa.tmp', header=None, sep=' ', 
                  skipinitialspace=True, 
                  names=['freq', 'power', 'lower', 'upper']
                  ).sort_index(by='freq')
plt.figure()
title = 'MSSA Singular Spectrum'
d.plot(x='freq', y='power', logy=True, linewidth=2, title=title, legend='power')
d.plot(x='freq', y='lower', logy=True, linewidth=1, color='k', legend='95% CI red noise')
d.plot(x='freq', y='upper', logy=True, linewidth=1, color='k')
plt.ylabel('Eigenvalues**1/2 (power)')
plt.xlabel('Frequency (cycles/year)')

### 3) plot EOFs
d = pd.read_table(fname+'_spc.tmp_mssaeofs.tmp', header=None, sep=' ', 
                  skipinitialspace=True)
d[d.columns[0:10]].plot(subplots=True, legend=False, linewidth=2, 
                        title='MSSA EOFs', yticks=[], xlim=[0, 77])


### 4) plot PCs
d = pd.read_table(fname+'_spc.tmp_mssapcs.tmp', header=None, sep=' ', 
                  skipinitialspace=True)
d.plot(subplots=True, linewidth=2, legend=False, title='MSSA PCs (Modes)',
       yticks=[], xlim=[d.index[0],d.index[-1]])

plt.show()


