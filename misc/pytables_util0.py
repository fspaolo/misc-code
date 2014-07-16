'''
I agree that pytable lack a really simple interface. Say something that
dumps a dic to an hdf5 file, and vice-versa (althought hdf5 -> dic is a
bit harder as all the hdf5 types may not convert nicely to python types).

On my experiment I use this code to load the data:
'''

def load_h5(file_name):
    """ Loads an hdf5 file and returns a dict with the hdf5 data in it.
    """
    file = tables.openFile(file_name)
    out_dict = {}
    for key, value in file.leaves.iteritems():
        if isinstance(value, tables.UnImplemented):
            continue
        try:
            value = value.read()
            try:
                if isinstance(value, CharArray):
                    value = value.tolist()
            except Exception, inst:
                print "Couldn't convert %s to a list" % key
                print inst
            if len(value) == 1:
                value = value[0]
            out_dict[key[1:]] = value
        except Exception, inst:
            print "couldn't load %s" % key
            print inst
    file.close()
    return(out_dict)

'''
It works well on our files, but our files are produced by code I wrote,
so they do not explore all the possibilities of hdf5.

Similarily I have some python code to dump a dic of arrays to an hdf5
file:
'''

def dic_to_h5(filename, dic):
    """ Saves all the arrays in a dictionary to an hdf5 file.
    """
    out_file = tables.openFile(filename, mode = "w")
    for key, value in dic.iteritems():
        if isinstance( value, ndarray):
           out_file.createArray('/', str(key), value)
    out_file.close()

'''
This code is not general enough to go in pytables, but if the list wants
to improve it a bit, then we could propose it for inclusion, or at least
put it on the cookbook.

Cheers,

Gaël
'''

'''
As I said before, be used to recarrays. If you have reasons for sticking
with dictionaries, it is straighforward converting a dict into a
recarray. For example:
'''

>>> v1=numpy.random.rand(10,)
>>> v2=numpy.random.randint(10, size=10)
>>> mydict={'v1':v1,'v2':v2}

#Conversion to a recarray begins

>>> cols = [col for col in mydict.itervalues()]
>>> ratype = [(name, col.dtype) for (name, col) in mydict.iteritems()]
>>> ra=numpy.rec.fromarrays(cols, dtype=ratype)

# now, you can proceed to saving (and reading) the data

>>> tra3=f.createTable('/', 'ra3', ra)
>>> tra3[:]
array([(0.71896141583591389, 3), (0.6147395923362261, 8),
       (0.74390300993242819, 8), (0.85740583591803832, 8),
       (0.058988577053635471, 4), (0.33839332688847212, 9),
       (0.3847836118934358, 2), (0.0072535131033339972, 5),
       (0.42023038711482563, 5), (0.26398728887523382, 6)],
      dtype=[('v1', '<f8'), ('v2', '<i4')])

'''
Yeah, there are infinite possibilities in that regard. However, I think
that there is a beauty in keeping the values of a dictionary (or fields
in a recarray) tied together in a table. This approach has proven to be
very powerful in many situations (but, of course, the user has to decide
the better way to arrange his own data).

Cheers,

Francesc Altet
'''
