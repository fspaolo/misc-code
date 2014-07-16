# I don't know of any built-in way to do this, 
# but a wrapper function is easy enough to write:

def read_in_chunks(infile, chunk_size=1024*64):
    while True:
        chunk = infile.read(chunk_size)
        if chunk:
            yield chunk
        else:
            # The chunk was empty, which means we're at the end
            # of the file
            return

# Then at the interactive prompt:

>>> from chunks import read_in_chunks
>>> infile = open('quicklisp.lisp')
>>> for chunk in read_in_chunks(infile):
...     print chunk
... 
<contents of quicklisp.lisp in chunks>

# Of course, you can easily adapt this to use a with block:

with open('quicklisp.lisp') as infile:
    for chunk in read_in_chunks(infile):
        print chunk

# And you can eliminate the if statement like this.

def read_in_chunks(infile, chunk_size=1024*64):
    chunk = infile.read(chunk_size)
    while chunk:
        yield chunk
        chunk = infile.read(chunk_size)
