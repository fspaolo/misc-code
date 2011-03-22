#!/usr/bin/env python
"""
 Generate (and submit) batch files for (X)grid computing. 

 Example::

   $ python batch.py -s -j jobname -c '/path/to/prog -a arg' /path/to/files/*.ext 

 Fernando <fpaolo@ucsd.edu>
 March 6, 2011
"""

import argparse as ap
import plistlib as pl
import os
import sys

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='file[s] to process '
                    '[ex: /path/to/files/*.ext]')
parser.add_argument('-j', dest='jobname', default='job', help='a symbolic '
                    'name of the job [default: job]')
parser.add_argument('-o', dest='batchfile', default=None, help='name of '
                    'output batch file [default: <jobname>.xml]')
parser.add_argument('-e', dest='email', default='', help='email to send '
                    'notification of job state')
parser.add_argument('-s', dest='submit', default=False, action='store_const', 
                    const=True, help='submit batch file after creation')
parser.add_argument('-c', dest='execute', help="command (w/args) to execute "
                    "[ex: -c '/path/to/prog -a arg']")

args = parser.parse_args()
files = args.file
jobname = args.jobname
batchfile = args.batchfile
email = args.email
submit = args.submit
execute = args.execute
#jobid = args.jobid
if not batchfile: 
    batchfile = jobname + '.xml'

# define the the structure of an Xgrid batch.
plist = {
    'name': jobname,
    'notificationEmail': email,
    'taskSpecifications': {
        '#': { 
            'command': execute,
         }
    }
}

pl.writePlist(plist, batchfile)
            
job = \
"""
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" 
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
<dict>
\t<!-- Job -->
\t<key>name</key>
\t<string>%s</string>
\t<!-- Send notifications -->
\t<key>notificationEmail</key>
\t<string>%s</string>
\t<!-- Task specifications -->
\t<key>taskSpecifications</key>
\t<dict>
%s
\t</dict>
</dict>
</array>
</plist>
"""
task = \
"""
\t\t<!-- Task -->
\t\t<key>%d</key>
\t\t<dict>
\t\t\t<!-- Command to execute -->
\t\t\t<key>command</key>
\t\t\t<string>%s</string>
\t\t\t<!-- Arguments -->
\t\t\t<key>arguments</key>
\t\t\t<array>
%s              
\t\t\t</array>
\t\t</dict>
"""

'''
def create_plist(execute, files):
    execute = execute.split()
    command, args = execute[0], execute[1:]
    tasks = ''
    for i, f in enumerate(files):
        argsfile = args + [f]
        arguments = ''.join(['\t\t\t\t<string>%s</string>\n' % arg for arg in argsfile])
        tasks = ''.join([tasks, task % (i, command, arguments)])
    return job % (jobname, email, tasks)

def create_batch(plist, batchfile):
    fout = open(batchfile, 'w')
    fout.write(plist)
    fout.close()
    print 'batch file ->', batchfile 

def submit_batch():
    conname = os.environ.get('XGRID_CONTROLLER_HOSTNAME')
    conpwd = os.environ.get('XGRID_CONTROLLER_PASSWORD')
    if (not conname) or (not conpwd):
        print 'error submitting: no controller name/password specified!'
        sys.exit()
    print 'job submitted:'
    os.system('xgrid -job batch %s > %s.id' % (batchfile, jobname))
    os.system('cat %s.id' % jobname)
    print 'job ID -> %s.id' % jobname 

def get_results(jobid):
    os.mkdir(jobname)
    os.system('xgrid -job resutlts -id %d -so %s.out -e %s.err -out %s' \
              % (jobid, jobname, jobname, jobname))
    print 'std output -> %s.out' % jobname
    print 'std error -> %s.err' % jobname
    print 'results dir -> %s' % jobname

plist = create_plist(execute, files)
create_batch(plist, batchfile)
if submit: submit_batch()
'''

#if jobid is not None: get_results(jobid)
