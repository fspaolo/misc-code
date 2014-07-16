# -*- coding: utf-8 -*-

# Laplace filter a 2D field with mask

# ROMS type maske, 
#        M = 1 på grid-celler som brukes
#        M = 0 på grid-celler som maskeres vekk
# Default er uten maske

import numpy as np


def laplace_X(F,M):
    """1D Laplace Filter in X-direction (axis=1)"""

    jmax, imax = F.shape

    # Add strips of land
    F2 = np.zeros((jmax, imax+2), dtype=F.dtype)
    F2[:, 1:-1] = F
    M2 = np.zeros((jmax, imax+2), dtype=M.dtype)
    M2[:, 1:-1] = M

    MS = M2[:, 2:] + M2[:, :-2]
    FS = F2[:, 2:]*M2[:, 2:] + F2[:, :-2]*M2[:, :-2]

    return np.where(M > 0.5, (1-0.25*MS)*F + 0.25*FS, F)

def laplace_Y(F,M):
    """1D Laplace Filter in Y-direction (axis=1)"""

    jmax, imax = F.shape

    # Add strips of land
    F2 = np.zeros((jmax+2, imax), dtype=F.dtype)
    F2[1:-1, :] = F
    M2 = np.zeros((jmax+2, imax), dtype=M.dtype)
    M2[1:-1, :] = M

    MS = M2[2:, :] + M2[:-2, :]
    FS = F2[2:, :]*M2[2:, :] + F2[:-2, :]*M2[:-2, :]

    return np.where(M > 0.5, (1-0.25*MS)*F + 0.25*FS, F)


# The mask may cause laplace_X and laplace_Y to not commute
# Take average of both directions

def laplace_filter(F, M=None):
    if M == None:
        M = np.ones_like(F)
    return 0.5*(laplace_X(laplace_Y(F, M), M) +
                laplace_Y(laplace_X(F, M), M))







    
                    

