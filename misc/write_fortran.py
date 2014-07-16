import numpy as np

# create a matrix of doubles
#data = np.arange(100, dtype='f8').reshape(10,10)
data = np.arange(10, dtype='f8')

# create a binary file
f = open('file.bin', 'wb')

# write data as a string of bytes in Fortran order
f.write(data.tostring(order='F')) 
