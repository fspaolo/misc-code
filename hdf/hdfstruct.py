#!/usr/bin/python

#    A usefull python program that will display the structure of 
#    the SD component of any HDF4 file whose name is given on the 
#    command line. After the HDF4 file is opened, high level 
#    inquiry methods are called to obtain dictionnaries descrybing 
#    attributes, dimensions and datasets. The rest of the program 
#    mostly consists in nicely formatting the contents of those 
#    dictionaries.

import sys
from pyhdf.SD import *

# Dictionary used to convert from a numeric data type to its 
# symbolic representation
typeTab = {
           SDC.CHAR:    'char',
           SDC.CHAR8:   'char8',
           SDC.UCHAR8:  'uchar8',
           SDC.INT8:    'int8',
           SDC.UINT8:   'uint8',
           SDC.INT16:   'int16',
           SDC.UINT16:  'uint16',
           SDC.INT32:   'int32',
	       SDC.UINT32:  'uint32',
           SDC.FLOAT32: 'float32',
           SDC.FLOAT64: 'float64'
	       }

printf = sys.stdout.write

def eol(n=1):
    printf("%s" % chr(10) * n)
    
hdfFile = sys.argv[1]    # Get first command line argument

# Catch pyhdf.SD errors
try:  
  # Open HDF file named on the command line
  f = SD(hdfFile)
  # Get global attribute dictionnary
  attr = f.attributes(full=1)
  # Get dataset dictionnary
  dsets = f.datasets()

  # File name, number of attributes and number of variables.
  printf("FILE INFO"); eol(2)
  #printf("-------------"); eol()
  printf("%-25s%s" % ("  file:", hdfFile)); eol()
  printf("%-25s%d" % ("  file attributes:", len(attr))); eol()
  printf("%-25s%d" % ("  datasets:", len(dsets))); eol()
  eol();

  # Global attribute table.
  if len(attr) > 0:
      printf("FILE ATTRIBUTES"); eol(2)
      printf("  name                 idx type    len value"); eol()
      printf("  -------------------- --- ------- --- -----"); eol()
      # Get list of attribute names and sort them lexically
      attNames = attr.keys()
      attNames.sort()
      for name in attNames:
          t = attr[name]
              # t[0] is the attribute value
              # t[1] is the attribute index number
              # t[2] is the attribute type
              # t[3] is the attribute length
          printf("  %-20s %3d %-7s %3d %s" %
                 (name, t[1], typeTab[t[2]], t[3], t[0])); eol()
      eol()


  # Dataset table
  if len(dsets) > 0:
      printf("DATASETS"); eol(2)
      printf("  name                 idx type    na cv dimension(s)"); eol()
      printf("  -------------------- --- ------- -- -- ------------"); eol()
      # Get list of dataset names and sort them lexically
      dsNames = dsets.keys()
      dsNames.sort()
      # sort according the last pos of the values (the index)
      for name in dsNames:
          # Get dataset instance
          ds = f.select(name)
          # Retrieve the dictionary of dataset attributes so as
          # to display their number
          vAttr = ds.attributes()
          t = dsets[name]
              # t[0] is a tuple of dimension names
              # t[1] is a tuple of dimension lengths
              # t[2] is the dataset type
              # t[3] is the dataset index number
          printf("  %-20s %3d %-7s %2d %-2s " %
                 (name, t[3], typeTab[t[2]], len(vAttr),
                  ds.iscoordvar() and 'X' or ''))
	  # Display dimension info.
          n = 0
          for d in t[0]:
              printf("%s%s(%d)" % (n > 0 and ', ' or '', d, t[1][n]))
              n += 1
          eol()
      eol()

  ''' ### not useful
  # Dataset info.
  if len(dsNames) > 0:
      printf("DATASET INFO"); eol()
      printf("-------------"); eol(2)
      for name in dsNames:
          # Access the dataset
          dsObj = f.select(name)
	  # Get dataset attribute dictionnary
          dsAttr = dsObj.attributes(full=1)
          if len(dsAttr) > 0:
              printf("%s attributes" % name); eol(2)
              printf("  name                 idx type    len value"); eol()
              printf("  -------------------- --- ------- --- -----"); eol()
	      # Get the list of attribute names and sort them alphabetically.
              attNames = dsAttr.keys()
              attNames.sort()
              for nm in attNames:
                  t = dsAttr[nm]
                      # t[0] is the attribute value
                      # t[1] is the attribute index number
                      # t[2] is the attribute type
                      # t[3] is the attribute length
                  printf("  %-20s %3d %-7s %3d %s" %
                         (nm, t[1], typeTab[t[2]], t[3], t[0])); eol()
              eol()
	  # Get dataset dimension dictionnary
          dsDim = dsObj.dimensions(full=1)
	  if len(dsDim) > 0:
	      printf ("%s dimensions" % name); eol(2)
              printf("  name                 idx len   unl type    natt");eol()
	      printf("  -------------------- --- ----- --- ------- ----");eol()
	      # Get the list of dimension names and sort them alphabetically.
	      dimNames = dsDim.keys()
	      dimNames.sort()
	      for nm in dimNames:
	          t = dsDim[nm]
		      # t[0] is the dimension length
		      # t[1] is the dimension index number
		      # t[2] is 1 if the dimension is unlimited, 0 if not
		      # t[3] is the the dimension scale type, 0 if no scale
		      # t[4] is the number of attributes
		  printf("  %-20s %3d %5d  %s  %-7s %4d" %
		         (nm, t[1], t[0], t[2] and "X" or " ", 
			  t[3] and typeTab[t[3]] or "", t[4])); eol()
	      eol()
  '''
  printf("note: idx:index #, na:# attributes, cv:coord var"); eol()
      
except HDF4Error, msg:
    print "HDF4Error", msg
