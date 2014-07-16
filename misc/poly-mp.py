import numpy
from multiprocessing import Pool
from time import time
import subprocess, os, sys

N = 1000*1000*10
expr = ".25*x**3 + .75*x**2 - 1.5*x - 2"  # the polynomial to compute
#expr = "((.25*x + .75)*x - 1.5)*x - 2"   # a computer-friendly polynomial
#expr = "numpy.sin(x)**2+numpy.cos(x)**2"             # a transcendental function
xp = numpy.linspace(-1, 1, N)

parallel = True
NT = 2

global result
result = numpy.empty(N, dtype='float64')

global counter
counter = 0
def cb(r):
    global counter
    global result
    #print r, counter
    y, nt, i = r     # unpack return code
    result[i*N/nt:(i+1)*N/nt] = y   # assign the correct chunk

def compute(nt, i):
    x = xp[i*N/nt:(i+1)*N/nt]
    y = eval(expr)
    return y, nt, i

if __name__ == '__main__':

    print "Serial computation..."
    t0 = time()
    res = compute(1,0)
    print "result (serial) -->", res[0]
    ts = round(time() - t0, 3)
    print "Time elapsed in serial computation:", ts

    if not parallel:
        sys.exit()

    for nt in range(2, NT+1):
        t0 = time()
        po = Pool(processes=nt)
        for i in xrange(nt):
            po.apply_async(compute, (nt,i), callback=cb)
        po.close()
        po.join()
        tp = round(time() - t0, 3)
        print "Time elapsed in parallel computation:", tp, "with %s threads" % nt

        print "Speed-up: %sx" % round(ts/tp, 2)
        
        #print "Result (parallel) -->", result
