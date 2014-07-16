#!/usr/bin/env python
"""
 Upload files to a server using the SSH protocol.

 Examples
 --------
 To upload 'file1.ext' and 'file2.ext'::

 $ python upload.py -u username -s server.ucsd.edu:/some/dir file1.ext file2.ext

 To upload several PDF files using the default <user> and <server>::
  
 $ python upload.py *.pdf

 Notes
 -----
 The default values are: 
 
    username=helen
    server=glaciology2.ucsd.edu:/var/www/html/sio115/

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# April 3, 2011

#import argparse as ap
import sys
import os

SERVER = 'glaciology2.ucsd.edu:/var/www/html/sio115/'

### to upload at igppweb.ucsd.edu/~helen comment out the following:
#SERVER = 'igppgateway.ucsd.edu:~/Sites/'    

# parse command line arguments
'''
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='files to upload')
parser.add_argument('-u', dest='user', default='helen', 
                    help="username [default: helen]")  
parser.add_argument('-s', dest='server', default=None, help="ex: "
                    "glaciology2.ucsd.edu:/var/www/html/sio115/ [default]")  
args = parser.parse_args()

USER = args.user
if args.server is not None:
    SERVER = args.server
'''
if len(sys.argv) < 2:
    print 'no input files found!'
    sys.exit()

files = sys.argv[1:]
#USER = 'helen'
USER = 'fpaolo'

#for f in args.files:
for f in files:
    #os.system('rsync -rav -e ssh %s %s@%s' % (f, USER, SERVER))
    os.system('scp -C %s %s@%s' % (f, USER, SERVER))
