#!/usr/bin/env python

import os
import gmtpy
from sys import exit

model = 'grav_modelcs_res.txt'
inputdir = '/home/fspaolo/DATA/work/norte/grav_ref6m'
region = '308/330/-6/6'
filter='g25'

txtgrid = os.path.join(inputdir, model)
bingrid = os.path.join(inputdir, model[:-4] + '.bin')
binfilt = os.path.join(inputdir, model[:-4] + '_filt.bin')
txtfilt = os.path.join(inputdir, model[:-4] + '_filt.txt')

gmt = gmtpy.GMT()

gmt.xyz2grd(G=bingrid, D='degree/degree/mGal/1/0/=/=', R=region, I='1m',
    V=True, out_discard=True, in_filename=txtgrid) #suppress_defaults=True

gmt.grdfilter(G=binfilt, F=filter, D=2, V=True, out_discard=True, suppress_defaults=True, in_filename=bingrid) 

gmt.grd2xyz(in_filename=binfilt, out_discard=True, out_filename=txtfilt)
