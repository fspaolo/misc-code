from __future__ import division
import numpy as np
cimport numpy as np

ctypedef np.int16_t SHORT

#cimport cython
#@cython.boundscheck(False)  # turn of bounds-checking for entire function
def mask_search(np.ndarray[SHORT, ndim=1] x,
                np.ndarray[SHORT, ndim=1] y,
                np.ndarray[SHORT, ndim=1] x_mask,
                np.ndarray[SHORT, ndim=1] y_mask,
                np.ndarray[SHORT, ndim=2] m_mask,
                np.ndarray[SHORT, ndim=2] flags,
                int R):

    cdef int x_i, y_i, f, out
    cdef unsigned int i, row, col, r, c
    cdef int N = x.shape[0]

    if x is None or y is None or x_mask is None or y_mask is None or \
       m_mask is None or flags is None:
        raise ValueError('error: mask_search: array set to None')

    for i in range(N):
        x_i = x[i]
        y_i = y[i]

        # if data point within mask limits
        if x_mask[0]+R < x_i and x_i < x_mask[-1]-R and \
           y_mask[0]+R < y_i and y_i < y_mask[-1]-R:

            row, = np.where(y_mask == y_i)
            col, = np.where(x_mask == x_i)
            f =  m_mask[row,col]          # 0=land/1=water/2=ice-shelf
            flags[i,0] = f 

            # neighboring values on a square 2Rx2R -> border flag: 0/1
            if np.alltrue(m_mask[<unsigned int>(row-R):<unsigned int>(row+R+1), \
                                 <unsigned int>(col-R):<unsigned int>(col+R+1)] == f):
                flags[i,1] = 0    # if all True
            else:                                          
                flags[i,1] = 1    # else is border

            # just another way of doing the main loop (C style)
            #out = 0
            #for r in range(row-R, row+R+1):
            #    for c in range(col-R, col+R+1):
            #        if m_mask[r,c] != f:
            #            flags[i,1] = 1
            #            out = 1
            #            break
            #    if out == 1: break
