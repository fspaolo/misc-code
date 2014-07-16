"""
This is the main user interface module, but most of the user
documentation, and the normal user API, is in pytide.__init__().
"""

import os
import sys
import struct
import numpy as np

from tidepredict import mjd_from_date, mjd_from_dday
from tidepredict import mapfuncs
from _tidesubs import ptide, def_cid


head, tail = os.path.split(os.path.realpath(__file__))

default_model = 'tpxo7.2'
default_dir = os.path.abspath(os.path.join(head, '../../tide'))

_minor_true = np.array(1, dtype="i") # True to infer minor constituents
_minor_false = np.array(0, dtype="i")

class Bunch(dict):
    """
    A dictionary that also provides access via attributes.

    There is an extended version in pycurrents.system, but this is
    all we need here.
    """
    def __init__(self, *args, **kwargs):
        """
        *args* can be dictionaries, bunches, or sequences of
        key,value tuples.  *kwargs* can be used to initialize
        or add key, value pairs.
        """
        dict.__init__(self)
        self.__dict__ = self
        for arg in args:
            self.update(arg)
        self.update(kwargs)


class TPXO_model_base(object):
    ilatshift = dict(h=0.5, u=0.5, v=0, z=0.5)
    jlonshift = dict(h=0.5, u=0, v=0.5, z=0.5)

    def _set_minfrac(self, mf):
        if mf is None:
            self._minfrac = 0.5
        elif mf > 0 and mf <=1.0:
            self._minfrac = mf
        else:
            raise ValueError("minfrac out of range: 0 < minfrac <= 1")

    def _get_minfrac(self):
        return self._minfrac

    minfrac = property(_get_minfrac, _set_minfrac)

    def _set_cid(self, c_idstring):
        self.cid = np.fromstring(c_idstring, dtype="S1")
        self.cid.shape = (len(c_idstring)//4, 4)
        self._ind = def_cid(self.cid)
        self.constituents = [c.strip()
                             for c in self.cid.view(dtype='|S4').flat]

    def make_grids(self, lonmin, lonmax, latmin, latmax):
        dx = (lonmax - lonmin) / float(self.nlon)
        dy = (latmax - latmin) / float(self.mlat)
        self.dx = dx
        self.dy = dy
        self.lonmin = lonmin
        self.latmin = latmin
        self.lonmax = lonmax
        self.latmax = latmax
        self.lonh = lonmin + (np.arange(self.nlon) + 0.5) * dx
        self.lath = latmin + (np.arange(self.mlat) + 0.5) * dy
        self.latu = self.lath
        self.lonv = self.lonh
        self.lonu = self.lonh - 0.5 * dx
        self.latv = self.lath - 0.5 * dy
        self.lon_periodic = abs(lonmax - lonmin - 360) < 0.001

    def xy_to_ij_float(self, x, y, var):
        i = (y - self.latmin) / self.dy - self.ilatshift[var]
        j = (x - self.lonmin) / self.dx - self.jlonshift[var]
        if self.lon_periodic:
            j %= self.nlon
        return i, j


    def ij_to_weights(self, i, j, goodmask):
        """
        Calculate bilinear interpolation weights and indices for a
        set of points with coordinates i, j in index space.

        i, j may be floating point sequences of the same length, N.

        goodmask is a 2-D array, True where data are good.

        Returns:

            *w* : 4xN array of weights
            *frac* : array of N floats, fraction of cell contributing data.
            *i0, j0, i1, ji* : sequences of opposite corner indices for
                interpolation rectangle

        """
        #i0 = np.floor(i).astype(int)
        #np.clip(i0, 0, self.mlat-2, out=i0)
        #j0 = np.floor(j).astype(int)
        #np.clip(j0, 0, self.nlon-2, out=j0)

        i0raw = np.floor(i).astype(int)
        i0 = np.clip(i0raw, 0, self.mlat-2)
        j0raw = np.floor(j).astype(int)
        j0 = np.clip(j0raw, 0, self.nlon-2)
        badmask = (i0 != i0raw) | (j0 != j0raw)

        i1 = i0 + 1
        j1 = j0 + 1
        gm = np.array([goodmask[i0, j0], goodmask[i0, j1],
                       goodmask[i1, j0], goodmask[i1, j1]]).astype(np.bool)

        di = i % 1
        dj = j % 1

        w = np.array([(1-di) * (1-dj), (1-di) * dj,
                          di * (1-dj),    di  * dj], dtype=np.float_)
        w *= gm

        frac = w.sum(axis=0)
        frac[badmask] = 0
        valid = frac > 0     # Only need to prevent division by zero here.
        w[:, valid] /= frac[:,valid]  # w[:, valid].sum(axis=0)
        w[:, ~valid] = 0
        return w, frac, i0, j0, i1, j1

    def interp_depth(self, x, y):
        v = self.z
        i, j = self.xy_to_ij_float(x, y, 'z')
        goodmask = self.zmask
        w, frac, i0, j0, i1, j1 = self.ij_to_weights(i, j, goodmask)
        vv = np.array([v[i0, j0], v[i0, j1],
                       v[i1, j0], v[i1, j1]], dtype=v.dtype)
        vv *= w
        varout = vv.sum(axis=0)
        return varout, frac

    def _ilist_from_clist(self, clist):
        ilist = [self.constituents.index(c) for c in clist]
        return ilist

    def _ptide(self, constit, t, clist=None):
        if clist is None:
            return ptide(constit, self.cid, self._ind, t, _minor_true)
        if len(clist) == 0:
            return ptide(constit, self.cid, self._ind, t, _minor_false)
        ilist = self._ilist_from_clist(clist)
        return ptide(constit[ilist], self.cid[ilist],
                        self._ind[ilist], t, _minor_false)

    def _badx(self, x, y):
        badx = (y > (self.latmax - 0.5 * self.dy)) | (y < self.latmin)
        if not self.lon_periodic:
            # This may be applying more margin than necessary,
            # but it should not hurt in practice.
            badx |= ((x < (self.lonmin + 0.5*self.dx))
                        | (x > (self.lonmax - 1.5*self.dx)))
        return np.ma.filled(badx, True)

    def _process_args(self, args):
        if len(args) == 1:
            arg = args[0]
            try:
                yearbase = arg.yearbase
                t = arg.dday
                x = arg.lon
                y = arg.lat
            except AttributeError:
                raise ValueError(
                "Single argument must have attributes yearbase, dday, lon, lat")
        elif len(args) == 2:
            yearbase = args[0]
            txy = np.asanyarray(args[1])
            t = txy[:,0]
            x = txy[:,1]
            y = txy[:,2]
        elif len(args) == 4:
            yearbase = args[0]
            t, x, y = map(np.atleast_1d, args[1:])
        else:
            raise ValueError(
                "Signatures require 1, 2, or 4 arguments")
        if x.shape != y.shape:
            raise ValueError("x, y must have same shape")

        if x.shape != t.shape and x.size != 1:
            raise ValueError(
                "with more than one position, times array must match positions")

        #FIXME: add more stringent argument checking?

        # Copy x and y to prevent side effects; np.ma.filled, used
        # below, does not make a copy if there are no masked elements.
        x = x.copy()
        y = y.copy()

        # This generates a new array for t as well:
        t = mjd_from_dday(yearbase, t)

        if self.grid_type == 0:
            x[x < self.lonmin] += 360
            x[x > self.lonmax] -= 360
        else:
            x, y = mapfuncs[self.grid_type](x, y)
        badx = self._badx(x, y)
        # Make x, y into ndarrays with bad values filled with locations
        # that are on land or off the grid.
        if self.grid_type == 0:
            x = np.ma.filled(x, 80)
            y = np.ma.filled(y, 60)
            x[badx] = 80
            y[badx] = 60
        else:
            x = np.ma.filled(x, 1e38)
            y = np.ma.filled(y, 1e38)
            x[badx] = 1e38
            y[badx] = 1e38
        return t, x, y, badx


    def _h_from_txy(self, t, x, y, badx, clist=None):
        constit, frac = self.interp_constit(x, y, 'h')
        cid = self.cid
        ind = self._ind
        _minor = _minor_true
        if clist is not None:
            _minor = _minor_false
            if len(clist) > 0:
                ilist = self._ilist_from_clist(clist)
                constit = constit[:,ilist]
                cid = self.cid[ilist]
                ind = self._ind[ilist]
        h = np.zeros(t.shape, dtype=np.float_)
        for i in range(t.shape[0]):
            isl = slice(i, i+1)
            h[i] = ptide(constit[i], cid, ind, t[isl], _minor)
        bad = frac <= self.minfrac
        badmask = bad | badx
        return np.ma.array(h, mask=badmask, copy=False), frac

    def _h_from_xy_times(self, t, x, y, badx, clist=None):
        if badx:
            ret = np.ma.zeros(t.shape, dtype=np.float)
            ret.mask = True
            return ret
        constit, frac = self.interp_constit(x, y, 'h')
        h = self._ptide(constit[0], t, clist=clist)
        badmask = (frac[0] <= self.minfrac) or self._badx(x, y)[0]
        return np.ma.array(h, mask=badmask, copy=False), frac

    def _uv_from_txy(self, t, x, y, badx, clist=None):
        uconstit, ufrac = self.interp_constit(x, y, 'u')
        vconstit, vfrac = self.interp_constit(x, y, 'v')
        cid = self.cid
        ind = self._ind
        _minor = _minor_true
        if clist is not None:
            _minor = _minor_false
            if len(clist) > 0:
                ilist = self._ilist_from_clist(clist)
                uconstit = uconstit[:,ilist]
                vconstit = vconstit[:,ilist]
                cid = self.cid[ilist]
                ind = self._ind[ilist]
        dep, dfrac = self.interp_depth(x, y)
        u = np.zeros(t.shape, dtype=np.float_)
        v = np.zeros(t.shape, dtype=np.float_)
        for i in range(t.shape[0]):
            isl = slice(i, i+1)
            u[i] = ptide(uconstit[i], cid, ind, t[isl], _minor)
            v[i] = ptide(vconstit[i], cid, ind, t[isl], _minor)
        frac = np.minimum(ufrac, vfrac)
        frac = np.minimum(frac, dfrac)
        valid = frac > 0
        u[valid] = u[valid] / dep[valid]
        v[valid] = v[valid] / dep[valid]
        badmask = (frac <= self.minfrac) | badx
        u = np.ma.array(u, mask=badmask, copy=False)
        v = np.ma.array(v, mask=badmask, copy=False)
        dep = np.ma.array(dep, mask=(frac < 0.01) | badx, copy=False)
        return u, v, dep, frac

    def _uv_from_xy_times(self, t, x, y, badx, clist=None):
        if badx:
            ret = np.ma.zeros(t.shape, dtype=np.float)
            ret.mask = True
            return ret, ret, np.zeros_like(ret), np.zeros_like(ret)
        dep, dfrac = self.interp_depth(x, y)
        constit, ufrac = self.interp_constit(x, y, 'u')
        u = self._ptide(constit[0], t, clist=clist)
        constit, vfrac = self.interp_constit(x, y, 'v')
        v = self._ptide(constit[0], t, clist=clist)
        frac = min(dfrac[0], ufrac[0])
        frac = min(frac, vfrac[0])
        valid = frac > 0
        if valid:
            u /= dep
            v /= dep
        badmask = (frac < self.minfrac) or self._badx(x, y)[0]
        u = np.ma.array(u, mask=badmask, copy=False)
        v = np.ma.array(v, mask=badmask, copy=False)
        return u, v, dep, frac

    def velocity(self, *args, **kw):
        """
        Calculate tide velocity components along a track or at a point.

        Arguments:
            txy_object
            yearbase, txy
            yearbase, t, x, y

        kwarg:
            clist

        In the first signature, txy_object is an object with
        attributes "yearbase", "dday", "lon", and "lat".

        In the second signature, txy is an Nx3 ndarray with columns
        decimal day, lon, lat.

        In the third signature, x and y must be the same length, and
        either of length 1 or the length of t.  In the former case,
        a time series is calculated at a single point.

        This method may be expanded later to handle a time series
        at an array of points.

        Returns: Bunch (dictionary and attribute access) with
        u, v, depth, fraction.
        depth is the model depth.
        fraction is 1 if the point is inside a full cell (with data
        at all 4 corners), tapering to zero for a point surrounded
        by land.  Estimates from cells with fraction < 1 should be
        considered highly suspect.

        """

        clist = kw.get("clist", None)
        t, x, y, badx = self._process_args(args)
        if len(x) == 1 and len(t) > 1:
            u, v, dep, frac = self._uv_from_xy_times(t, x, y, badx,
                                                            clist=clist)
        else:
            u, v, dep, frac = self._uv_from_txy(t, x, y, badx, clist=clist)
        return Bunch(u=u, v=v, depth=dep, fraction=frac)

    def height(self, *args, **kw):
        """
        Calculate tide height along a track or at a point.

        Arguments:
            txy_object
            yearbase, txy
            yearbase, t, x, y

        kwarg:
            clist

        In the first signature, txy_object is an object with
        attributes "yearbase", "dday", "lon", and "lat".

        In the second signature, txy is an Nx3 ndarray with columns
        decimal day, lon, lat.

        In the third signature, x and y must be the same length, and
        either of length 1 or the length of t.  In the former case,
        a time series is calculated at a single point.

        This method may be expanded later to handle a time series
        at an array of points.

        Returns: Bunch (dictionary and attribute access) with
        h, fraction.
        fraction is 1 if the point is inside a full cell (with data
        at all 4 corners), tapering to zero for a point surrounded
        by land.  Estimates from cells with fraction < 1 should be
        considered highly suspect.

        """

        clist = kw.get("clist", None)
        t, x, y, badx = self._process_args(args)
        if len(x) == 1 and len(t) > 1:
            h, frac = self._h_from_xy_times(t, x, y, badx,
                                                            clist=clist)
        else:
            h, frac = self._h_from_txy(t, x, y, badx, clist=clist)
        return Bunch(h=h, fraction=frac)


class TPXO_model(TPXO_model_base):
    """
    Work with the original OSU Fortran model files.
    """

    def __init__(self, model=None, modeldir=None, minfrac=None):
        """
        *model* is the OSU model file name without the "Model_",
        e.g., 'tpxo7.2'
        This file has 3 or 4 lines, one each for the h, u, and grid file
        names.  All 3 of these model-related files must be in the same
        directory, given by *modeldir*.  If present, the 4th line
        is the name of a map conversion function, indicating that
        the model grid is a map projection rather than a lat/lon grid.
        """
        if model is None:
            model = default_model
        if modeldir is None:
            modeldir = default_dir
        self.modeldir = modeldir
        self.model = model
        self._set_minfrac(minfrac)
        p = os.path.join(modeldir, "Model_" + model)
        fns = [line.strip() for line in open(p).readlines()]

        # Strip away the directory:
        fns = [os.path.split(line)[-1] for line in fns]
        # Strip off .nc if present; this is a kluge to allow
        # a testing directory with both nc and fortran data files,
        # so we can use the nc fortran code to generate test data.
        fns = [fn.split(".nc")[0] for fn in fns]

        self.h_fname = os.path.join(modeldir, fns[0])
        self.u_fname = os.path.join(modeldir, fns[1])
        self.d_fname = os.path.join(modeldir, fns[2])

        self.grid_type = 0 # standard: lat lon grid
        #
        if len(fns) > 3:
            modfunc = fns[3]
            # Odd numbers for north, even for south.
            # 1 and 2 from old OTIS code.
            if modfunc.startswith("xy_ll_N"):
                self.grid_type = 1
            elif modfunc.startswith("xy_ll_S"):
                self.grid_type = 2
            # Old OTIS code does not support the CATS2008 from ERS:
            elif modfunc.startswith("xy_ll_CATS"):
                self.grid_type = 4

        h_file = open(self.h_fname)
        nbrec, n, m, nc = struct.unpack(">4I", h_file.read(16))
        self.nlon = n
        self.mlat = m
        self.nc = nc

        latmin, latmax, lonmin, lonmax = struct.unpack(">4f",
                                           h_file.read(16))
        self.make_grids(lonmin, lonmax, latmin, latmax)


        c_idstring = h_file.read(nc*4)
        h_file.close()
        self._set_cid(c_idstring)
        dt = np.dtype(">c8")
        self.vars = dict()
        self.h = []
        for i in range(nc):
            ofs = nbrec + 12 + i * (8 + 8 * m * n)
            self.h.append(np.memmap(self.h_fname, offset=ofs, mode='r',
                            dtype=dt, shape=(m, n)))
        dt = np.dtype([("u", ">c8"), ("v", ">c8")])
        self.u = []
        self.v = []
        for i in range(nc):
            ofs = nbrec + 12 + i * (8 + 16 * m * n)
            uv = np.memmap(self.u_fname, offset=ofs, mode='r',
                             dtype=dt, shape=(m, n))
            self.u.append(uv["u"])
            self.v.append(uv["v"])

        self.read_grid()

    def read_grid(self):
        f = open(self.d_fname)
        recsize, n, m = struct.unpack(">3I", f.read(12))
        f.seek(8 + recsize)
        rec2size = struct.unpack(">I", f.read(4))[0]
        f.close()
        z_off = 12 + recsize + 8 + rec2size
        m_off = z_off + 8 + 4*m*n
        self.z = np.memmap(self.d_fname, offset=z_off, mode='r',
                             dtype=">f4", shape=(m, n))
        self.zmask = np.memmap(self.d_fname, offset=m_off, mode='r',
                             dtype=">I", shape=(m,n))
        # zmask masks the Caspian and Black Seas, and a few other spots
        # where z is not zero.  It could be eliminated by multiplying it
        # times z.


    def write_binary(self):
        """
        Write out a single-file version of the model for
        efficient access via the TPXO_model_uh class.
        """
        if sys.byteorder != 'little':
            raise RuntimeError(
                "write_binary is implemented only for little-endian machines")
        fname = 'uh_le_%s' % self.model
        out = open(os.path.join(self.modeldir, fname), 'w')
        out.write(struct.pack("<4I", self.grid_type,
                                    self.mlat, self.nlon, self.nc))
        out.write(struct.pack("<4f", self.latmin, self.latmax,
                                        self.lonmin, self.lonmax))
        self.cid.tofile(out)

        z = np.empty((self.mlat, self.nlon + 1), dtype=np.float32)
        z[:, :self.nlon] = self.z
        if self.lon_periodic:
            z[:, self.nlon] = self.z[:, 0]
        else:
            z[:, self.nlon] = 0
        z.tofile(out)
        del(z)

        zm = np.empty((self.mlat, self.nlon + 1), dtype=np.bool8)
        zm[:, :self.nlon] = self.zmask
        if self.lon_periodic:
            zm[:, self.nlon] = self.zmask[:, 0]
        else:
            zm[:, self.nlon] = 0
        zm.tofile(out)
        del(zm)

        h = np.empty((self.mlat, self.nlon + 1, self.nc), dtype=np.complex64)
        for var in [self.h, self.u, self.v]:
            for i in range(self.nc):
                h[:, :self.nlon, i] = var[i]
            if self.lon_periodic:
                h[:, self.nlon, :] = h[:, 0, :]
            else:
                h[:, self.nlon, :] = 0
            h.tofile(out)

        out.close()



    def interp_constit(self, x, y, var):
        v = getattr(self, var)
        i, j = self.xy_to_ij_float(x, y, var)
        goodmask = v[0]
        w, frac, i0, j0, i1, j1 = self.ij_to_weights(i, j, goodmask)
        varout = np.zeros((len(x), self.nc), dtype=v[0].dtype)
        for k, vin in enumerate(v):
            vv = np.array([vin[i0, j0], vin[i0, j1],
                           vin[i1, j0], vin[i1, j1]], dtype=vin.dtype)
            vv *= w
            varout[:,k] = vv.sum(axis=0)
        return varout, frac


class TPXO_model_uh(TPXO_model_base):
    """
    Access the efficient single-file form of the model.
    """

    def __init__(self, model=None, modeldir=None, minfrac=None):
        """
        model is the file name without the prefix,
        e.g., 'tpxo7.2'
        """
        if model is None:
            model = default_model
        if modeldir is None:
            modeldir = default_dir
        if sys.byteorder != 'little':
            raise RuntimeError(
                "TPXO_model_uh is only for little-endian machines")
        self.modeldir = modeldir
        self.model = model
        self._set_minfrac(minfrac)
        self.fname = os.path.join(modeldir, "uh_le_%s" % model)

        tp_file = open(self.fname)
        gt, m, n, nc = struct.unpack("4I", tp_file.read(16))
        self.grid_type = gt
        self.nlon = n
        self.mlat = m
        self.nc = nc

        latmin, latmax, lonmin, lonmax = struct.unpack("4f",
                                           tp_file.read(16))
        self.make_grids(lonmin, lonmax, latmin, latmax)

        c_idstring = tp_file.read(nc*4)
        ofs = tp_file.tell()
        tp_file.close()

        self._set_cid(c_idstring)

        self.z = np.memmap(self.fname, offset=ofs, mode='r',
                                dtype=np.float32, shape=(m, n+1))
        ofs += m * (n+1) * 4

        self.zmask = np.memmap(self.fname, offset=ofs, mode='r',
                                dtype=np.bool8, shape=(m, n+1))
        ofs += m * (n+1)

        self.h = np.memmap(self.fname, offset=ofs, mode='r',
                                dtype=np.complex64, shape=(m, n+1, nc))
        ofs += m * (n+1) * nc * 8

        self.u = np.memmap(self.fname, offset=ofs, mode='r',
                                dtype=np.complex64, shape=(m, n+1, nc))
        ofs += m * (n+1) * nc * 8

        self.v = np.memmap(self.fname, offset=ofs, mode='r',
                                dtype=np.complex64, shape=(m, n+1, nc))
        ofs += m * (n+1) * nc * 8



    def interp_constit(self, x, y, var):
        v = getattr(self, var)
        i, j = self.xy_to_ij_float(x, y, var)
        goodmask = v[:,:,0]
        w, frac, i0, j0, i1, j1 = self.ij_to_weights(i, j, goodmask)

        vv = np.array([v[i0, j0], v[i0, j1],
                       v[i1, j0], v[i1, j1]], dtype=v.dtype)
        vv *= w[..., np.newaxis]
        varout = vv.sum(axis=0)
        return varout, frac

    def write_binary(self):
        """
        Dummy method; warn that we are already using the single file.
        """
        warnings.warn(
            "Single file model is already in use; no need to write_binary()")


def model(name=None, modeldir=None, writemodel=False, minfrac=None):
    """
    Factory function to return a tide model access class instance.

    *name* is the OSU model file name without the "Model_",
        e.g., 'tpxo7.2' (default)

    *modeldir* is the directory in which the model files may
        be found, defaulting to '../../tide' relative to the
        pytide directory itself.

    *writemodel* : if True, an optimized model file will be written
        if it does not already exist; default is False.

    *minfrac* : if not None (default), values will be interpolated
        only if the sum of the weights on surrounding points is
        greater than or equal to the specified number, before
        normalization; the maximum value is 1, which means the point
        is within a rectangle of valid points.

    Model defaults are pytide module attributes, *default_model*, and
    *default_dir*.

    *minfrac* is a property, so it can be changed after initialization.

    Returns a TPXO_model_uh instance if available; otherwise returns
    a TPXO_model instance.  Data files for the former may be generated
    from the latter, so as to be available for future use.
    """
    if sys.byteorder == 'little':
        try:
            return TPXO_model_uh(name, modeldir, minfrac=minfrac)
        except IOError:
            pass
        if writemodel:
            TPXO_model(name, modeldir).write_binary()
            return TPXO_model_uh(name, modeldir, minfrac=minfrac)
    return TPXO_model(name, modeldir, minfrac=minfrac)


##################################################################

# Below this point, functions are for local testing rather than
# for general use.  Some require things that are not in the
# pytide repo.

def read_test_data():
    a = np.fromfile(open('lat_lon_time'), sep=" ")
    a = a.reshape((len(a)//8, 8))
    lat = a[:,0]
    lon = a[:,1]
    date = a[:,2:]
    return lat, lon, date

def read_test_out(fname):
    from pycurrents.codas import to_day
    lines = open(fname).readlines()[6:]
    dt = np.dtype([('t', np.float),
                   ('x', np.float),
                   ('y', np.float),
                   ('U', np.float),
                   ('V', np.float),
                   ('u', np.float),
                   ('v', np.float),
                   ('d', np.float)])
    dat = np.zeros((len(lines),), dtype=dt)
    for i, line in enumerate(lines):
        fields = line.split()
        dat['x'][i] = fields[1]
        dat['y'][i] = fields[0]
        dat['U'][i] = fields[4]
        dat['V'][i] = fields[5]
        dat['u'][i] = fields[6]
        dat['v'][i] = fields[7]
        dat['d'][i] = fields[8]
        mdy = map(int, fields[2].split('.'))
        hms = map(int, fields[3].split(':'))

        dat['t'][i] = to_day(mdy[2], mdy[2], mdy[0], mdy[1], *hms)

    return dat

setup_template = """DATA/Model_tpxo7.2         ! 1. tidal model control file
lat_lon_time.tmp           ! 2. latitude/longitude/time file
u                          ! 3. z/U/V/u/v
                           ! 4. tidal constituents to include
AP                         ! 5. AP/RI
oce                        ! 6. oce/geo
1                          ! 7. 1/0 correct for minor constituents
test_out.tmp               ! 8. output file (ASCII)
"""

def run_original(txy, yearbase):
    """
    Note the hardwired location of the original predict_tide executable.
    This also requires that there be an original-style DATA directory
    in (or symlinked into) the current working directory.
    """
    from pycurrents.codas import to_date
    open("setup.tmp", "w").write(setup_template)
    lines = []
    ymdhms = to_date(yearbase, txy[:,0])
    for i, (t, x, y) in enumerate(txy):
        args = tuple([y, x] + list(ymdhms[i]))
        lines.append("%f %f %d %d %d %d %d %d\n" % args)
    open("lat_lon_time.tmp", 'w').writelines(lines)
    # continue by running the file
    os.system("/home/manini/programs/pytide/predict_tide < setup.tmp")

def compare_original(txy, yearbase):
    """
    Compare uv_from_txy method output to the original Fortran.

    Note the restrictions described in the run_original() docstring.
    """
    import matplotlib.pyplot as plt
    run_original(txy, yearbase)
    dat = read_test_out("test_out.tmp")
    mod = TPXO_model_uh()
    u, v, dep, frac = mod.uv_from_txy(txy, yearbase)
    plt.plot(dat['t'], dat['u'], 'ko',
             dat['t'], u*100, 'g+',
             dat['t'], dat['u']-u*100, 'r.')
    plt.show()
    return u, v, dep, frac, dat



