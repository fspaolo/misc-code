from fabric.api import *

SEASAT_work     ='/Volumes/LaCie1TB/ANTARCTICA/SEASAT/work'
GEOSAT_GM_work  ='/Volumes/LaCie1TB/ANTARCTICA/GEOSAT_GM/work'
GEOSAT_ERM_work ='/Volumes/LaCie1TB/ANTARCTICA/GEOSAT_ERM/work'
GFO_work        ='/Volumes/LaCie1TB/ANTARCTICA/GFO/work'
ERS1_work       ='/Volumes/LaCie1TB/ANTARCTICA/ERS1/work'
ERS2_work       ='/Volumes/LaCie1TB/ANTARCTICA/ERS2/work'
ENVISAT_work    ='/Volumes/LaCie1TB/ANTARCTICA/ENVISAT/work'

XG_BATCH = 'python xg-batch.py'
PYTHON =  '/Library/Frameworks/EPD64.framework/Versions/Current/bin/python'

def timesep(month=None):
    if month is None:
        EXECUTE = '-c "%s /Users/fpaolo/code/separate/timesep.py"' % PYTHON

        JOBNAME = '-j year-seasat -s'
        FILES = SEASAT_work + '/*_bias.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j year-geogm -s'
        FILES = GEOSAT_GM_work + '/*_bias.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j year-geoerm -s'
        FILES = GEOSAT_ERM_work + '/*_bias.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        #JOBNAME = '-j year-gfo -s'
        #FILES = GFO_work + '/*_bias.h5'
        #local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j year-ers1 -s'
        FILES = ERS1_work + '/*_bias.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j year-ers2 -s'
        FILES = ERS2_work + '/*_bias.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j year-envisat -s'
        FILES = ENVISAT_work + '/*_bias.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))
    else:
        EXECUTE = '-c "%s /Users/fpaolo/code/separate/timesep.py -m"' % PYTHON

        #JOBNAME = '-j month-seasat -s'
        #FILES = SEASAT_work + '/*_bias_????.h5'
        #local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j month-geogm -s'
        FILES = GEOSAT_GM_work + '/*_bias_????.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j month-geoerm -s'
        FILES = GEOSAT_ERM_work + '/*_bias_????.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        #JOBNAME = '-j month-gfo -s'
        #FILES = GFO_work + '/*_bias_????.h5'
        #local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j month-ers1 -s'
        FILES = ERS1_work + '/*_bias_????.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j month-ers2 -s'
        FILES = ERS2_work + '/*_bias_????.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

        JOBNAME = '-j month-envisat -s'
        FILES = ENVISAT_work + '/*_bias_????.h5'
        local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

def merge():
    CMD =  '/Users/fpaolo/code/misc/merge2.py'    # [edit]
    EXECUTE = ' '.join(['-c', '"', PYTHON, CMD, '"'])
    
    #JOBNAME = '-j merge-seasat -s -1'    # [edit]
    #FILES = SEASAT_work + '/*_????_??.h5'    # [edit]
    #local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

    JOBNAME = '-j merge-geogm -s -1'    # [edit]
    FILES = GEOSAT_GM_work + '/*_????_??.h5'    # [edit]
    local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

    JOBNAME = '-j merge-geoerm -s -1'    # [edit]
    FILES = GEOSAT_ERM_work + '/*_????_??.h5'    # [edit]
    local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

    #JOBNAME = '-j merge-gfo -s -1'    # [edit]
    #FILES = GFO_work + '/*_????_??.h5'    # [edit]
    #local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

    JOBNAME = '-j merge-ers1 -s -1'    # [edit]
    FILES = ERS1_work + '/*_????_??.h5'    # [edit]
    local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

    JOBNAME = '-j merge-ers2 -s -1'    # [edit]
    FILES = ERS2_work + '/*_????_??.h5'    # [edit]
    local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))

    JOBNAME = '-j merge-envisat -s -1'    # [edit]
    FILES = ENVISAT_work + '/*_????_??.h5'    # [edit]
    local(' '.join([XG_BATCH, JOBNAME, EXECUTE, FILES]))
