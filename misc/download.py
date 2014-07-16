#!/usr/bin/env python
"""
  download.py

  Simple program to download a whole bunch of files
  from a web page.

  Fernando Paolo - Jul-2008.
  (modified from Prahbu Ramachandran)
"""

import urllib, re
import os
from sys import argv, exit

#-------------------------------------------------------------------
url = 'http://www.lithoflex.org/listing/USP/Course_sept2008/'
ext = 'ppt'
dir = 'teste'
#-------------------------------------------------------------------


def check_arg(argv):
    """Check arguments and files."""
    if len(argv) < 3:  # check args
        print 'Usage: python %s <url> <extension>' % argv[0]
        exit()
    else:
	url = argv[1]
        ext = argv[2]
    if not path.exists(filein):  # check dir
        print 'Input file not found:' 
        print filein
        exit()
    return url, ext


def get_page(url):
    """Open a URL."""
    return urllib.urlopen(url)


def get_name(url):
    """Get file name at the end of a URL."""
    i = 1
    while url[-i] != "/" and i < len(url):
	i += 1
    return url[-i+1:]


def get_file(url, file):
    """Download <url> and save it in <file>."""
    os.mkdir(dir)
    dst = os.path.join(dir, filename)
    urllib.urlretrieve(url, dst)


def read_page(url):
    """Read the entire page to scan it."""
    page = get_page(url)
    data = page.read()
    return data


def get_all(data, base=None, extn=ext):
    """Scan page and get all files ended by <ext>."""
    patn = re.compile(r'HREF="([^"]+?\.%s)"' % extn, re.I)
    tmp = patn.findall(data)
    if base:
        files = []
        for file in tmp:
            files.append(urllib.basejoin(base, file))
        return files
    else:
        return tmp


def dwnld_all(data, base=None, extn=ext):
    """Download all files ended by <ext> to a local dir"""
    files = get_all(data, base, extn)
    for file in files:
        filename = get_name(file)
	# Downloading one by one.
        print file
	get_file(file, filename)


def main():
    #url, ext = check_arg(argv)
    data = read_page(url)
    dwnld_all(data, url, ext)
    

if __name__ == "__main__":
    main()
