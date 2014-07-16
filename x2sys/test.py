import os
import sys
import re
import numpy as np
from itertools import combinations as comb

files = sys.argv[1:]

date = lambda s: re.findall('\d\d\d\d\d\d', s)
comb2 = lambda f: [ff for ff in comb(f, 2)]

def remove_duplicates(pairs):
    for i, f in enumerate(pairs):
        if ('_asc' in f[0] and '_asc' in f[1]) or \
           ('_des' in f[0] and '_des' in f[1]) or \
           date(f[0]) == date(f[1]):
            pairs[i] = 'x' 
    return [ff for ff in pairs if ff != 'x'] 

pairs = comb2(files)
pairs = remove_duplicates(pairs)

'''
for f in pairs:
    print f
'''

print 'number of pairs of files:', len(pairs)
