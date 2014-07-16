import numpy as np
import tables as hdf
import sys, os

# orbit, utc, lat, lon, elev, f2, f4, f7, f8, f10, f14, f15, f31
#     0    1    2    3     4   5   6   7   8    9   10   11   12

jf2 = 5
jf4 = 6
jf7 = 7
jf8 = 8
jf10 = 9
jf14 = 10
jf15 = 11
jf31 = 12

files = sys.argv[1:]
if len(files) < 1 or '-h' in files:
    print 'usage: python %s infiles.h5' % sys.argv[0]
    sys.exit()

print 'filtering', len(files), 'files ...'

nfine = 0
nmedium = 0
ncoarse = 0
nother = 0
npoints = 0
for f in files:
    h5file = hdf.openFile(f)
    data = h5file.root.data.read()
    h5file.close()
    
    N = len(data)

    # f2: wvfm spec shaped: 0=no, 1=yes
    if len(data) > 1:
        f2 = data[:,jf2] 
        nf2 = len(f2[f2==1])

    # f4: wvfm spec retracked: 0=no, 1=yes
    if len(data) > 1:
        f4 = data[:,jf4] 
        nf4 = len(f4[f4==1])

    # f7: problem w/leading edge: 0=no, 1=yes	(probelm for Geosat) !!!
    if len(data) > 1:
        f7 = data[:,jf7] 
        nf7 = len(f7[f7==1])

    # f8: problem retracking: 0=no, 1=yes	(problem for Geosat) !!!
    if len(data) > 1:
        f8 = data[:,jf8] 
        nf8 = len(f8[f8==1])

    # f10: wvfm retracked: 0=no, 1=yes		(problem for Geosat) !!!
    if len(data) > 1:
        f10 = data[:,jf10] 
        nf10 = len(f10[f10==1])

    # Envisat: tracking mode (flags 14 and 15):
    # 0 0 = Fine (~ocean in ERS)
    # 0 1 = Medium (~ice in ERS)
    # 1 0 = Coarse (no in ERS)
    if len(data) > 1:
        f14 = data[:,jf14] 
        f15 = data[:,jf15] 
        ifine, = np.where((f14==0) & (f15==0))
        imedium, = np.where((f14==0) & (f15==1))
        icoarse, = np.where((f14==1) & (f15==0))
        iother, = np.where((f14==1) & (f15==1))
        nfine += len(ifine)   
        nmedium += len(imedium)   
        ncoarse += len(icoarse)
        nother += len(iother)


    npoints += N
    #break


#print 'IDR file         ', f
print 'number of points:', npoints 
print
#print 'flag 2 set:      ', nf2
#print 'flag 4 set:      ', nf4
#print 'flag 7 set:      ', nf7
#print 'flag 8 set:      ', nf8
#print 'flag 10 set:     ', nf10 

print 'Flags      14 15 (tracking mode)'
perc = ((np.float(nfine) / npoints) * 100)
print 'Fine mode   0  0: %d (%.2f%%)' % (nfine, perc)
perc = ((np.float(nmedium) / npoints) * 100)
print 'Medium mode 0  1: %d (%.2f%%)' % (nmedium, perc)
perc = ((np.float(ncoarse) / npoints) * 100)
print 'Coarse mode 1  0: %d (%.2f%%)' % (ncoarse, perc)
perc = ((np.float(nother) / npoints) * 100)
print 'Other       1  1: %d (%.2f%%)' % (nother, perc)
