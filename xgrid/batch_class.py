#!/usr/bin/env python

import argparse as ap
import sys
import os

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='files to process')
parser.add_argument('-e', dest='execute', 
    help='command to execute [ex: -e "prog -a arg"]')

args = parser.parse_args()
files = args.file
execute = args.execute

plist = \
"""
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
<dict>
	<!-- Job -->
	<key>name</key>
	<string>%s</string>
	<!-- Task specifications -->
	<key>taskSpecifications</key>
	<dict>
%s
	</dict>
</dict>
</array>
</plist>
"""

class Job(object):

    def __init__(self, name):
        self.name = name
        self.job = plist % (name, '%s')

    def _command(self, cmd):
        return ''.join([
            '\t\t\t<!-- Command to execute -->\n',
            '\t\t\t<key>command</key>\n',
            '\t\t\t<string>%s</string>\n' % cmd,
            ])

    def _arguments(self, args, file):
        args = args + [file]
        s1 = ''.join([
            '\t\t\t<!-- Arguments -->\n',
            '\t\t\t<key>arguments</key>\n',
            '\t\t\t<array>\n',
            ])
        s2 = ''.join(['\t\t\t<string>%s</string>\n' % arg for arg in args])
        s3 = '\t\t\t</array>\n'
        return ''.join([s1, s2, s3])

    def _task(self, num, cmd, args):
        return ''.join([
            '\t\t<!-- Task -->\n',
            '\t\t<key>%d</key>\n' % num,
            '\t\t<dict>\n',
            cmd,
            args,
            '\t\t</dict>\n',
            ])

    def tasks(self, execute, files):
        execute = execute.split()
        cmd, args = execute[0], execute[1:]
        ntasks = ''
        for i, f in enumerate(files):
            icmd = self._command(cmd)
            iargs = self._arguments(args, f)
            itask = self._task(i, icmd, iargs)
            ntasks = ''.join([ntasks, itask])
        self.job = self.job % ntasks 

j = Job('fernando')
j.tasks(execute, files)
print j.job
