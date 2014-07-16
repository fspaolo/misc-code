#!/usr/bin/env python

"""
Converts xyz point cloud into something renderable
Original code by Rod Holland & Prabhu Ramachandran
Handy to visualise raw ascii column file

feel free to simplify the textfile interactions. 
it doesn't really require scientific/numeric to 
be installed to do what it does.
"""

#~ from Scientific.IO.TextFile import TextFile
#~ from Numeric import *
import sys
import string
from sys import argv


if len(argv)>1: 
        filename = argv[1]
        output = str(filename[:-3])+"vtk"
        print "output to:", output
        
else :
        print """Usage: 'python xyz2vtk.py pointclouddatafile'
Converts xyz file into vtk UNSTRUCTURED GRID format
for point cloud rendering."""
        sys.exit(0)

                # change i for headers in ascii file
i=0
x=[]
y=[]
z=[]
f = open(filename,'r')
for line in f.readlines():
        words = string.split(line)
        i=i+1
        if i>0:
            if len(words)> 0:
                #for j in range(len(words)):
                    x.append(words[0])
                    y.append(words[1])
                    z.append(words[2])
            
n=len(x)
print "n:",n
                    # write data to vtk polydata file
                    # write header
out = open(output, 'w')
h1 = """# vtk DataFile Version 2.0
loop
ASCII
DATASET UNSTRUCTURED_GRID
POINTS """ + str(n) + """ float
"""
out.write(h1)
                    # write xyz data
for i in range(n):
        #s = '%15.2f %15.2f %15.2f' % (x[i], y[i], z[i])
        out.write(str(x[i])+" "+str(y[i])+" "+str(z[i])+'\n')
        
                    # write cell data
out.write("CELLS "+ str(n)+ " "+ str(2*n)+'\n')
for i in range(n):
        #s = '1 %d \n' % (i)
        out.write("1 "+str(i)+"\n")
        
                    # write cell types
out.write("CELL_TYPES " + str(n)+'\n')
for i in range(n): out.write("1 \n")

                    # write z scalar values
h2 = '\n' + """POINT_DATA """ + str(n) + "\n" 
h3 = """SCALARS Z_Value float 1
LOOKUP_TABLE default"""
out.write(h2+ h3+'\n')
for i in range(n):
        sc=(z[i])
        out.write(str(sc)+ "\n")

                   
out.write('\n')
out.close()
