
file_in = '/dados3/fspaolo/DATA/norte/grav/models/grav_modelcs_2m_res.txt'
file_out = '/dados3/fspaolo/DATA/norte/grav/models/grav_modelcs_2m_res.txt.clean_err'

param_col = 3    # column of the parameter to be cleaned

max_val = 180  # grav_max = 100-150 mGal, ssg_max = 100-150 urad (1 urad ~ 1 mGal)

edit = 1         # 1 = yes, 0 = no (just show the results)

#-----------------------------------------------------------

import numpy as N

print 'loading data ...'
x = N.loadtxt(file_in)
row, col = x.shape

print '(before)'
print 'total pts:', row
print 'mean:', N.mean(x[:,param_col])
print 'max:', N.max(x[:,param_col])
print 'min:', N.min(x[:,param_col])

if edit == 1:

    y = N.empty((row, col), 'f')
    
    k = 0
    for i in xrange(row): 
    
        if N.abs(x[i,param_col]) <= max_val and x[i,param_col] >= 0.0:
            y[k,:] = x[i,:]
            y[k,param_col] = N.sqrt(y[k,param_col])
            k += 1
    
    N.savetxt(file_out, y[:k,:], fmt='%f', delimiter=' ')

    print '(after)'
    print 'rejected:', row-k
    print 'mean:', N.mean(y[:k,param_col])
    print 'max:', N.max(y[:k,param_col])
    print 'min:', N.min(y[:k,param_col])
    print 'output ->', file_out
