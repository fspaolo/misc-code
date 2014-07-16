import sys
import os

fname1 = '/Users/fpaolo/data/altim/ERS2/fris/1995/ERS2_1995_Aug_asc.txt' 
fname2 = '/Users/fpaolo/data/altim/ERS2/fris/1995/ERS2_1995_Dec_des.txt' 
fout = '/Users/fpaolo/data/altim/ERS2/fris/1995/xover.txt' 

os.system('./xover %s %s %s' % (fname1, fname2, fout))

'''
 in file: /Users/fpaolo/data/altim/ERS2/fris/1995/ERS2_1995_Aug_asc.txt                                       
 in file: /Users/fpaolo/data/altim/ERS2/fris/1995/ERS2_1995_Dec_des.txt                                       
 out file: /Users/fpaolo/data/altim/ERS2/fris/1995/xover.txt                                                   
 processing...
 
 done!

real	42m10.565s
user	41m52.953s
sys	0m6.236s 

icesat:1995 fpaolo$ cat xover.txt | wc -l
    7860

icesat:x2sys fpaolo$ time sh x2sys.sh 
Password:
x2sys_init: File exists: ALTIM/ALTIM.tag
x2sys_init: Removed file /opt/local/share/gmt/x2sys/ALTIM/ALTIM.tag
x2sys_init: File exists: ALTIM/altim.def
x2sys_init: Removed file /opt/local/share/gmt/x2sys/ALTIM/altim.def
x2sys_init: File exists: ALTIM/ALTIM_tracks.d
x2sys_init: Removed file /opt/local/share/gmt/x2sys/ALTIM/ALTIM_tracks.d
x2sys_init: File exists: ALTIM/ALTIM_paths.txt
x2sys_init: Removed file /opt/local/share/gmt/x2sys/ALTIM/ALTIM_paths.txt
x2sys_init: File exists: ALTIM/ALTIM_index.b
x2sys_init: Removed file /opt/local/share/gmt/x2sys/ALTIM/ALTIM_index.b
x2sys_init: Initialize ALTIM/ALTIM.tag
x2sys_init: Initialize ALTIM/altim.def
x2sys_init: Initialize ALTIM/ALTIM_tracks.d
x2sys_init: Initialize ALTIM/ALTIM_index.b
x2sys_init: Initialize ALTIM/ALTIM_paths.txt
x2sys_init completed successfully
x2sys_cross: Files found: 2
x2sys_cross: Checking for duplicates : 0 found
x2sys_cross: No time column, use dummy times
x2sys_cross: Processing /Users/fpaolo/data/altim/ERS2/fris/1995/ERS2_1995_Aug_asc - /Users/fpaolo/data/altim/ERS2/fris/1995/ERS2_1995_Dec_des : 46354

real	9m22.681s
user	8m52.880s
sys	0m1.771s

icesat:1995 fpaolo$ cat x2sys.txt | wc -l
    8888
'''
