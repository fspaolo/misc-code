import numpy as N

file1 = '/home/fspaolo/DATA/norte/geoid/geoid_modelcs_2m.txt'
file2 = '/home/fspaolo/DATA/norte/geoid/geoid_modelcs_2m_bat.txt'
fileout = '/home/fspaolo/DATA/norte/geoid/geoid_modelcs_2m_bat.txt'

a = N.loadtxt(file1)
b = N.loadtxt(file2)

x = N.column_stack((a,b[:,2]))



N.savetxt(fileout, x, fmt='%f', delimiter=' ')

print 'output ->', fileout

#--------------------------------------------------------

file1 = '/home/fspaolo/DATA/norte/geoid/geoid_modelcs_2m_bat.txt'
file2 = '/home/fspaolo/DATA/norte/geoid/geoid_modelcs_2m_egm.txt'
fileout = '/home/fspaolo/DATA/norte/diff/geoid_modelcs_2m_egm.txt'

bat = 3
grav = 2

a = N.loadtxt(file1)
b = N.loadtxt(file2)

m,n = a.shape

x = N.empty((m,n), 'f')

j = 0
for i in xrange(m):
    if a[i,bat] < 0.0:  # ocean
        x[j,:] = a[i,:]
        x[j,grav] -= b[i,grav]
        j += 1

N.savetxt(fileout, x[:j,:], fmt='%f', delimiter=' ')

print 'output ->', fileout
