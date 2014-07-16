"""
Filter selected flags.

Fernando Paolo <fpaolo@ucsd.edu>
November 15, 2010
"""

import numpy as np
import tables as tb
import sys
import os

"""
Data definition (see `readidr_ra1.f90`):

 0 orbit
 1 utc85
 2 lat
 3 lon
 4 elev
 5 agc
 6 fmode : tracking mode
     0 = ocean | fine
     1 = ice | medium
     2 = none | coarse
 7 fret : waveform was retracked
     0 = no
     1 = yes
 8 fprob : problem retracking            # don't use this for Geosat/GM
     0 = no (passed all the tests)
     1 = yes (didn't pass at least one test)
 9 fmask : MOA mask
     0 = land
     1 = water 
     2 = ice-shelf
10 fbord : is border
     0 = no
     1 = yes
11 ftrack : separate tracks
     0 = ascending
     1 = descending
     2 = invalid

"""

#------------------------------------------------------------------------
# edit here
#------------------------------------------------------------------------

# accept selected (key) values!
column = {'fmode': 6, 'fret': 7, 'fprob': 8, 'fmask': 9, 'fbord': 10, 'fsep':11}
flag = {'fmode': None, 'fret': 1, 'fprob': 0, 'fmask': 2, 'fbord': None, 'fsep' : None}  
ext = '_shelf.h5'

#------------------------------------------------------------------------

files = sys.argv[1:]
if len(files) < 1 or '-h' in files:
    print 'usage: python %s file1.h5 file2.h5 ...' % sys.argv[0]
    sys.exit()

print 'filtering files:', len(files), '...'

npoints = 0
nfiltered = 0
for f in files:
    fin = tb.openFile(f)
    #data = fin.root.data.read()  # in-memory
    data = fin.root.data          # out-of-memory
    
    npoints += data.shape[0] 

    if data.shape[0]:
        ind, = np.where( \
            #(np.int8(data[:, column['fmode']]) == flag['fmode']) & \
            (np.int8(data[:, column['fret']]) == flag['fret']) & \
            (np.int8(data[:, column['fprob']]) == flag['fprob']) & \
            (np.int8(data[:, column['fmask']]) == flag['fmask']) \
            #(np.int8(data[:, column['fbord']]) == flag['fbord']) \
            #(np.int8(data[:, column['fsep']]) == flag['fsep']) \
            )

        nfiltered += data.shape[0] - ind.shape[0]

        if ind.shape[0] > 0:
            fout = tb.openFile(os.path.splitext(f)[0] + ext, 'w')
            shape = data[ind,:].shape
            atom = tb.Atom.from_dtype(data.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                                     filters=filters)
            dout[:] = data[ind,:]
            fout.close()

    fin.close()

perc = ((np.float(nfiltered) / npoints) * 100)
nleft = npoints - nfiltered
print 'done!'
print 'total points:', npoints 
print 'filtered out: %d (%.1f%%)' % (nfiltered, perc)
print 'points left: %d (%.1f%%)' % (nleft, 100-perc)
