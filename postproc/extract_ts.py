import numpy as np
import tables as tb
import matplotlib.pyplot as plt

fn = tb.openFile('/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5')
fo = tb.openFile('/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5.old')
