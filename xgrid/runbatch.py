#!/usr/bin/env python

import os
import re
from glob import glob

class Xover(object):

    def __init__(self):
        self.file_ref = ''
        self.files_to_cross = ''
        self.job_name = ''
        self.email = ''

    def run_job(self):
        if 'envisat' in self.job_name:
            self.email = '-e fspaolo@gmail.com'
        else:
            self.email = '-e fspaolo@gmail.com'
        self._get_files_to_cross()
        self._generate_cmd()
        self._generate_and_send_job()

    def _get_files_to_cross(self):
        self.files = glob(self.files_to_cross)
        # sort files using dates. Very important !!!
        self.files.sort(key=lambda s: re.findall('\d\d\d\d\d\d', s))
        #self.files = self.files[2:]    # exclude first two
        if not self.files:
            raise IOError('no files to cross')

    def _generate_cmd(self):
        self.cmd = \
            '/Library/Frameworks/EPD64.framework/Versions/Current/bin/python ' \
            '/data/xgrid/code/x2sys/x2sys.py -r %s' % self.file_ref # ADD `-r` <<<<<<<<<
                          
    def _generate_and_send_job(self):
        os.system('python xg-batch.py %s -j %s -s -2 -c "%s" %s' 
            % (self.email, self.job_name, self.cmd, ' '.join(self.files)) )
    '''
    def _generate_and_send_job(self):
        os.system('python xg-batch.py %s -j %s -s -0 -c "%s" %s' 
            % (self.email, self.job_name, self.cmd, ' '.join(self.files)) )
    '''

x = Xover()

#x.file_ref = '/data/xgrid/envisat/raw/envisat_200209_fris'

x.job_name = 'x2sys-fris-ers1'
x.files_to_cross = '/data/xgrid/ers1/fris/*_??????_fris'
x.run_job()

x.job_name = 'x2sys-fris-ers2'
x.files_to_cross = '/data/xgrid/ers2/fris/*_??????_fris'
x.run_job()

x.job_name = 'x2sys-fris-envisat'
x.files_to_cross = '/data/xgrid/envisat/fris/*_??????_fris'
x.run_job()

x.job_name = 'x2sys-ross-ers1'
x.files_to_cross = '/data/xgrid/ers1/ross/*_??????_ross'
x.run_job()

x.job_name = 'x2sys-ross-ers2'
x.files_to_cross = '/data/xgrid/ers2/ross/*_??????_ross'
x.run_job()

x.job_name = 'x2sys-ross-envisat'
x.files_to_cross = '/data/xgrid/envisat/ross/*_??????_ross'
x.run_job()
