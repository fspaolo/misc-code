"""
Module with some high level utility functions.

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import os
import re
import numpy as np
import scipy as sp
import tables as tb
import datetime as dt


def poly_fit(x, y, n=1):
    """
    Fit a polynomial of order `n` to data points `x,y`.
    """
    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    coeff = np.polyfit(x, y, n)
    x_val = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeff, x_val)
    return x_val, y_fit, coeff[0]


def get_size(arr):
    """
    Get the size in MB of a Numpy or PyTables object.

    parameters
    ----------
    arr : 1D/2D Numpy or PyTables Array.
    """
    try:
        m, n = arr.shape
    except:
        m, n = arr.shape, 1
    nelem = m*n
    isize = arr.dtype.itemsize
    return (isize*nelem/1e6)


def check_if_can_load(data, max_size=512):
    """
    Check if PyTables Array can be loaded in memory.
    """
    if get_size(data) > max_size:
        msg = 'data is larger than %d MB, not loading in-memory!' \
            % max_size
        raise MemoryError(msg)
    else:
        return True


def check_if_need_to_save(data, max_size=128):
    """
    Check when data in memory need to be flushed on disk.
    """
    data = np.asarray(data)
    if get_size(data) > max_size:
        return True
    else:
        return False



