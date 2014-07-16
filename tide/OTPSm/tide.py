"""
Wrapper around the Fortran code for OTPS software.

http://www.coas.oregonstate.edu/research/po/research/tide/

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# May 29, 2012

import os
import sys
import argparse as ap
import configobj as co
import numpy as np
import tables as tb
import datetime as dt
import matplotlib.dates as mpl
from subprocess import call

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='*', help='HDF5/ASCII file[s] to read')
parser.add_argument('-y', dest='refyear', type=int, default=1985, 
    help='reference year for the time in seconds, default 1985')  
parser.add_argument('-k', dest='skiprows', type=int, default=7,
    help='skip the first SKIPROWS lines from output file, default 7')
parser.add_argument('-c', dest='cols', nargs='*', default=[4],
    help='which columns to output (-c 0 1 6 7 ..), default: 4')
parser.add_argument('-s', dest='suffix', default='_tide', 
    help='suffix for the output filename, default none')
parser.add_argument('-r', dest='remove', default=False, action='store_const', 
    const=True, help='remove temp files after processing, default no')  
parser.add_argument('-i', dest='control', default=False, action='store_const', 
    const=True, help='generate the control file: Model_*, default no')  
args = parser.parse_args()


class Input(object):
    """
    [TO EDIT] Handles the program input.

    - Get specific variables from specific file type and structure
    - Prepare specific input for the processing
    """
    def __init__(self):
        self.path = os.path.split(os.path.realpath(__file__))[0]

    def read_config_file(self):
        # search for config file in the same dir as this script
        fname_conf = os.path.realpath(__file__).replace('.py', '.conf')
        if not os.path.exists(fname_conf):
            raise IOError('configuration file not found: %s' % fname_conf)
        # parse config file arguments
        path = self.path
        conf = co.ConfigObj(fname_conf)
        self.elev_model = os.path.join(path, conf['model']['elev_model'])
        self.transp_model = os.path.join(path, conf['model']['transp_model'])
        self.bathy_grid = os.path.join(path, conf['model']['bathy_grid'])
        self.load_model = os.path.join(path, conf['model']['load_model'])
        self.convert_func = conf['model']['convert_func']
        self.variable = conf['setup']['variable']
        self.constit = conf['setup']['constit']
        self.output = conf['setup']['output']
        self.tide = conf['setup']['tide']
        self.correct = conf['setup']['correct']
        self.predict_tide = os.path.join(path, conf['exec']['predict_tide'])

    def gen_control_file(self, control=False):
        mod_path, mod_name = os.path.split(self.elev_model)
        self.fname_control = os.path.join(mod_path, 'Model_' + \
            os.path.splitext(mod_name)[0])
        if control:
            fid = open(self.fname_control, 'w')
            fid.write(self.elev_model + '\n')
            fid.write(self.transp_model + '\n')
            fid.write(self.bathy_grid + '\n')
            fid.write(self.convert_func + '\n')
            fid.write(self.load_model + '\n')
            fid.close()
            print 'control file generated:', self.fname_control
            sys.exit()

    def gen_setup_file(self, fname_in):
        filename = os.path.splitext(fname_in)[0] 
        self.fname_in = fname_in
        self.fname_setup = filename + '.inp'
        self.fname_llt = filename + '.llt'
        self.fname_tide = filename + '.tide'
        fid = open(self.fname_setup, 'w')
        fid.write(self.fname_control + '\n')
        fid.write(self.fname_llt + '\n')
        fid.write(self.variable + '\n')
        fid.write(self.constit + '\n')
        fid.write(self.output + '\n')
        fid.write(self.tide + '\n')
        fid.write(self.correct + '\n')
        fid.write(self.fname_tide + '\n')
        fid.close()

    def gen_llt_file(self, refyear=1985):
        self.fin = tb.openFile(self.fname_in)
        ##########################################
        data = self.fin.getNode('/data')
        if len(data.shape) != 2: 
            print 'no data in file:', self.fname_in
            return 1
        lon, lat, secs1, secs2 = data[:,0], data[:,1], data[:,4], data[:,5]
        t1 = self.sec2dt(secs1, since_year=refyear)
        t2 = self.sec2dt(secs2, since_year=refyear)
        lat = np.hstack((lat, lat))
        lon = np.hstack((lon, lon))
        time = np.vstack((t1, t2))
        ##########################################
        X = np.column_stack((lat, lon, time))
        np.savetxt(self.fname_llt, X, fmt='%f %f %d %d %d %d %d %d')
        return 0

    def remove_temps(self):
        try:
            os.remove(self.fname_setup)
            os.remove(self.fname_tide)
            os.remove(self.fname_llt)
        except:
            pass

    def print_files(self):
        print '-'*20
        print 'control file:  ', self.fname_control
        print 'input file:    ', self.fname_in
        print 'setup file:    ', self.fname_setup
        print 'lat/lon/t file:', self.fname_llt
        print 'tide file:     ', self.fname_tide

    def sec2dt(self, secs, since_year=1985):
        dt_ref = dt.datetime(since_year, 1, 1, 0, 0)
        #return np.asarray([dt_ref + dt.timedelta(seconds=s) for s in secs])
        dtimes = np.asarray([dt_ref + dt.timedelta(seconds=s) for s in secs])
        return np.array([(d.year, d.month, d.day, d.hour,
            d.minute, d.second) for d in dtimes])

    # DEPRECATED
    def secs_to_datetime(self, secs, since_year=1985):
        if np.ndim(secs) > 0:
            secs = np.asarray(secs)
            secs[np.isnan(secs)] = 0.    # just in case
        REF_EPOCH_IN_DAYS = mpl.date2num(dt.date(since_year, 1, 1))  
        FRAC_DAYS = secs / 86400.
        dates = mpl.num2date(REF_EPOCH_IN_DAYS + FRAC_DAYS)
        return np.array([(d.year, d.month, d.day, d.hour,
            d.minute, d.second) for d in dates])


class Output(object):
    """
    [TO EDIT] Handle the program output.

    - Output file type and structure
    - What variables to save
    """
    def __init__(self, fname_in, fname_tide, suffix, skiprows, usecols):
        self.fname_out = os.path.splitext(fname_in)[0] + suffix + '.h5'
        self.fname_tide = fname_tide
        self.suffix = suffix
        self.skiprows = skiprows
        self.usecols = usecols

    def get_tides(self, split=1):
        try:
            self.z = np.loadtxt(self.fname_tide, 
                skiprows=self.skiprows, usecols=self.usecols)
            if split > 1:
                # for xover files: z1 and z2
                self.z = np.column_stack(np.split(self.z, split)) 
        except:
            self.z = None
            print 'problem loading file:', self.fname_tide
            #raise  # shows the error for debugging

    def save_data(self, data, z):
        if z is None: return
        n = np.ndim(z)
        fout = tb.openFile(self.fname_out, 'w')
        shape = (data.shape[0], data.shape[1] + n)
        atom = tb.Atom.from_dtype(data.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        dout = fout.createCArray(fout.root,'data', atom=atom,
            shape=shape, filters=filters)
        dout[:,:-n] = data[:]
        dout[:,-n:] = z[:] 
        fout.close()


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 

#------------------------------------------------------------------------

def main(args):

    files_in = args.files
    skiprows = args.skiprows
    usecols = args.cols
    suffix = args.suffix
    remove = args.remove
    control = args.control
    refyear = args.refyear

    print 'files to process:', len(files_in)
    print 'time is seconds since:', refyear

    In = Input()
    In.read_config_file()
    In.gen_control_file(control)

    nfiles = 0
    ntides = 0
    for ifile in files_in:

        In.gen_setup_file(ifile)
        stat = In.gen_llt_file(refyear)
        if stat == 1: continue
        In.print_files()

        #-------------------------------------------------------------

        os.system('%s < %s' % (In.predict_tide, In.fname_setup))

        #-------------------------------------------------------------

        Out = Output(In.fname_in, In.fname_tide, suffix, skiprows, usecols) 
        Out.get_tides(split=2)
        Out.save_data(In.fin.root.data, Out.z)

        if Out.z is not None: 
            nfiles += 1
            ntides += Out.z.size

        if remove: 
            In.remove_temps()

        In.fin.close()

    close_files()
    print 'done.'
    print 'tide predictions:', ntides
    print 'files created:', nfiles
    try:
        print 'last output file:', Out.fname_out
    except:
        pass

if __name__ == '__main__':
    main(args)
