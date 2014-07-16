#!/usr/bin/env python
#**************************************************
#  strreplace.py
#
#  Replace a string in multiple files.
#**************************************************

import fileinput, glob, string, sys, os
from os.path import join

if len(sys.argv) < 2:
    print "Usage:  python  %s  <search_str>  <replace_str>  <dir>" % \
          os.path.basename(sys.argv[0])
    sys.exit(0)

stext = sys.argv[1]
rtext = sys.argv[2]
if len(sys.argv) ==4:
    path = join(sys.argv[3], "*")
else:
    path = "*"

print "Finding: " + stext + ", replacing with: " + rtext + " in: " + path

files = glob.glob(path)
for line in fileinput.input(files, inplace=1):
     lineno = 0
     lineno = string.find(line, stext)
     if lineno > 0:
         line = line.replace(stext, rtext)

     sys.stdout.write(line)
