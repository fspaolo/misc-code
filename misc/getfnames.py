#!/usr/bin/env python

import os
import sys
import re

files = sys.argv[1:]
str = ' '.join(files)

def get_month_from_name(fname):
    date = re.findall('\d\d\d\d\d\d\d\d', fname)[0]
    return date[4:6]

print ' '.join([f for f in files if (get_month_from_name(f) in ['01', '04', '07', '10'])])
