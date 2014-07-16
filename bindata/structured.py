#! /usr/bin/env python

# ***********************************************************************************
# * Copyright 2010 Paulo A. Herrera. All rights reserved.                           * 
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ***********************************************************************************

# **************************************************************
# * Example of how to use the high level gridToVTK function.   *
# * This example shows how to export a structured grid.        *
# **************************************************************

from evtk.hl import gridToVTK
import numpy as np
import random as rnd

import tables as tb
import pyproj as pj

f = tb.openFile('/Users/fpaolo/data/fris/xover/seasonal/ers1_19920608_19960606_t.h5')

x = f.root.fris.x_edges[:-1]
y = f.root.fris.y_edges[:-1]
dh_mean = f.root.fris.dh_mean[:]
z = np.arange(0, dh_mean.shape[2])

lat_ts = -71              # standard lat (true scale)
lon_0 = 0; lat_0 = -90    # proj center (NOT grid center!)
proj = pj.Proj(proj='stere', lat_ts=lat_ts, lon_0=lon_0, lat_0=lat_0)
xx, yy = np.meshgrid(x, y)
xx, yy = proj(xx, yy)

f.close()

# Dimensions
nx, ny, nz = len(x), len(y), len(z)
print 'nx, ny, nz:,', nx, ny, nz
'''
lx, ly = x[-1]-x[0], y[-1]-y[0]
dx, dy, dz = lx/nx, ly/ny, lz/nz

ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)
'''

# Coordinates
'''
X = np.arange(0, lx + 0.1*dx, dx, dtype='float64')
Y = np.arange(0, ly + 0.1*dy, dy, dtype='float64')
Z = np.arange(0, lz + 0.1*dz, dz, dtype='float64')
'''

x3 = np.zeros((ny, nx, nz))
y3 = np.zeros((ny, nx, nz))
z3 = np.zeros((ny, nx, nz))

'''
# We add some random fluctuation to make the grid
# more interesting
for k in range(nz):
    for j in range(nx):
        for i in range(ny):
            x3[i,j,k], y3[i,j,k] = proj(x[j], y[i])
            #x3[i,j,k] = x[j]
'''

for k in range(16):
    x3[:,:,k] = xx[:]
    y3[:,:,k] = yy[:]
    z3[:,:,k] = k

# Variables
'''
pressure = np.random.rand(ncells).reshape( (nx, ny, nz))
temp = np.random.rand(npoints).reshape( (nx + 1, ny + 1, nz + 1))
'''

print 'x3, y3, z3, dh_mean:', x3.shape, y3.shape, z3.shape, dh_mean.shape
gridToVTK("./structured", x3, y3, z3, cellData = {"dh_mean" : dh_mean}) #, pointData = {"temp" : temp})
