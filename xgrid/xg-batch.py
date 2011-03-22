#!/usr/bin/env python
"""
 Generate and submit batch files for (X)grid computing. 

 Example::

   $ python batch.py -s -j jobname -c '/path/to/prog -a arg' /path/to/files/*.ext 

 About Xgrid::

   http://www.apple.com/server/macosx/technology/xgrid.html

 Fernando <fpaolo@ucsd.edu>
 March 6, 2011
"""

import argparse as ap
import plistlib as pl
import os
import sys

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='*', help='file[s] to process '
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
if not args.batchfile: 
    args.batchfile = args.jobname + '.xml'

# define a basic structure for an Xgrid batch.
PLIST = {
    'jobSpecification': {
        'applicationIdentifier': 'com.apple.xgrid.cli',
        'name': args.jobname,
        'notificationEmail': args.email,
        'schedulerParameters': {'dependsOnJobs': []},
        'inputFiles': {},
        'taskSpecifications': {},
        }
    }
            
def create_tasks(args):
    execute = args.execute.split()
    cmd, arguments = execute[0], execute[1:]
    tasks = {}
    if args.files:
        for i, f in enumerate(args.files):
            tasks.setdefault(`i`, {'command': cmd, 
                                   'arguments': arguments + [f]})
    else:
        tasks.setdefault(`0`, {'command': cmd, 
                               'arguments': arguments})
    return tasks

def input_files(args):
    inputFiles = {}
    for f in args.files:
       if not os.path.isabs(f):
           file = open(f).read()
           inputFiles.setdefault(f, {'fileData': pl.Data(file), 
                                     'isExecutable': 'YES'})
    return inputFiles

def submit_batch(args):
    controller = os.environ.get('XGRID_CONTROLLER_HOSTNAME')
    password = os.environ.get('XGRID_CONTROLLER_PASSWORD')
    if (not controller) or (not password):
        print 'error submitting: no controller name/password specified!'
        sys.exit()
    print 'job submitted:'
    os.system('xgrid -job batch %s > %s.id' % (args.batchfile, args.jobname))
    os.system('cat %s.id' % args.jobname)
    print 'job ID -> %s.id' % args.jobname 


PLIST['jobSpecification']['taskSpecifications'] = create_tasks(args)
PLIST['jobSpecification']['inputFiles'] = input_files(args) 

pl.writePlist(PLIST, args.batchfile)

print 'batch file ->', args.batchfile 

'''
def get_results(jobid):
    os.mkdir(jobname)
    os.system('xgrid -job resutlts -id %d -so %s.out -e %s.err -out %s' \
              % (jobid, jobname, jobname, jobname))
    print 'std output -> %s.out' % jobname
    print 'std error -> %s.err' % jobname
    print 'results dir -> %s' % jobname

PLIST = create_PLIST(execute, files)
create_batch(PLIST, batchfile)
'''
if args.submit: 
    submit_batch(args)
