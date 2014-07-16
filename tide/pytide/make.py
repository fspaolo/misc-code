#!/usr/bin/env python

import os
os.system("f2py --fcompiler=gnu95 -c  _tidesubs.pyf tidesubs.f")


