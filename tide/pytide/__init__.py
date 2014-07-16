"""
pytide wraps Fortran code from OSU to provide python access to
TPXO-style models.

Two styles of model data are supported: the original Fortran
unformatted binary files, and a single binary file that can be
generated from the originals.  At present, only models that
are on regular lon/lat grids are supported.

Example: suppose you have the Antarctic Peninsula model files
unpacked into /home/efiring/programs/TPXO/DATA.  To generate
the more efficient binary format in the same location, do::

  import pytide
  pytide.model("AntPen", "/home/efiring/programs/TPXO/DATA").write_binary()

Then to extract the u and v components of the predicted tide
for a cruise track, where t, x, y are specified in a flat
ascii file, "a_np.gps", where time is in decimal days relative
to 2010, do::

  import numpy as np
  txy = np.loadtxt("a_np.gps")
  tidemod = pytide.model("AntPen", "/home/efiring/programs/TPXO/DATA")
  vel = tidemod.velocity(2010, txy)
  # Now you can access vel.u, vel.v, vel.depth, vel.fraction.
  # If t, x, y come as separate 1-d arrays, you can also use
  # vel = tidemod.velocity(2010, t, x, y)
  # Or if working with a CODAS database:
  from pycurrents.codas import get_txy
  txy_object = get_txy("adcpdb/aship")
  vel = tidemod.velocity(txy_object)

For height as well, the same three styles of input are available, e.g.::

  height = tidemod.height(txy_object)
  # Now access height.h, height.depth, height.fraction.

If you want to look at time series of u, v, h at a point, say
62S, 60W, for the first 31 days of the year, do::

  t = np.arange(0, 31.01, 0.02)
  vel = tidemod.velocity(2010, t, -60, -62)

The extraction methods accept a kwarg, "clist", which can
be an empty list to turn off the correction for minor constituents,
or it can be a list of the desired constituents, e.g.::

  height = tidemod.height(2010, t, -60, -62,
                              clist=['m2', 's2', 'o1', 'k1'])

The minor consituent correction is done only if no clist is
given, or clist is None.

Use of the write_binary() method is optional; pytide.model() will
return a class instance for accessing the original multi-file
Fortran model if the single file version is not found.

After using write_binary(), you must call pytide.model again to get
an instance of the class accessing the single-file version.

The default model and default model directory are stored in
the module attributes *pytide.default_model* and *pytide.default_dir*,
which may be changed before calling *pytide.model()* if one wants
to avoid specifying the model and directory as arguments.

Note: this module is using the same bilinear interpolation scheme
as the OTIS FORTRAN code from which it is derived, although the
interpolation is being done in Python rather than in the extension
code.  Pytide can reproduce the results of the FORTRAN
predict_tide (to the limited extent that testing has been done;
user beware).  Pytide provides an additional option, however, to
control the masking of locations that are not surrounded by
valid grid points.  The *minfrac* kwarg in model(), and the
property of the same name, can be set in the range 0 to 1,
including 1 but exclusive of 0.  A value of 1 means no prediction
will be made for points not surrounded by 4 valid values.  Smaller
values relax the constraint, so that 0.75, for example, barely
admits a point on the diagonal in the middle of a cell that is
missing one corner.

The TMD Matlab code produces slightly different results, particularly
when interpolating within incomplete grid cells.  OTIS, TMD, and
pytide with *minfrac* = None (present default) all yield predictions
in highly incomplete grid cells; at least for velocity, such
predictions may bear no relation to reality.  Therefore it seems
preferable to mask them out by using a minfrac value of at least
0.5, which is presently the default. Changing the default to a larger
value is under consideration.  Because a "fraction" attribute is
returned by the velocity() and height() methods, the user can experiment
with masking criteria without having to rerun the model.


"""

from read_tpxo_bin import model, TPXO_model, TPXO_model_uh
from read_tpxo_bin import default_model, default_dir

