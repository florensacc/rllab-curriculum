
# AUTO GENERATED. DO NOT CHANGE!
from ctypes import *
import numpy as np

class MJDATA(Structure):
    
    _fields_ = [
        ("nbuffer", c_int),
        ("buffer", POINTER(c_ubyte)),
        ("qpos", POINTER(c_double)),
        ("qvel", POINTER(c_double)),
        ("qacc", POINTER(c_double)),
        ("qfrc_ext", POINTER(c_double)),
        ("qvel_next", POINTER(c_double)),
        ("xpos", POINTER(c_double)),
        ("xquat", POINTER(c_double)),
        ("xmat", POINTER(c_double)),
        ("xanchor", POINTER(c_double)),
        ("xaxis", POINTER(c_double)),
        ("geom_xpos", POINTER(c_double)),
        ("geom_xmat", POINTER(c_double)),
        ("com", c_double * 3),
        ("cdof", POINTER(c_double)),
        ("cinert", POINTER(c_double)),
        ("cvel", POINTER(c_double)),
        ("cacc", POINTER(c_double)),
        ("cfrc_int", POINTER(c_double)),
        ("qfrc_bias", POINTER(c_double)),
        ("crb", POINTER(c_double)),
        ("qM", POINTER(c_double)),
        ("qLD", POINTER(c_double)),
        ("neq", c_int),
        ("eq_err", POINTER(c_double)),
        ("eq_v", POINTER(c_double)),
        ("eq_J", POINTER(c_double)),
        ("eq_JMi", POINTER(c_double)),
        ("eq_A", POINTER(c_double)),
        ("nlim", c_int),
        ("ncon", c_int),
        ("lim_adr", POINTER(c_int)),
        ("con_adr", POINTER(c_int)),
        ("con_pair", POINTER(c_int)),
        ("con_xpos", POINTER(c_double)),
        ("con_xmatT", POINTER(c_double)),
        ("con_friction", POINTER(c_double)),
        ("lc_dist", POINTER(c_double)),
        ("lc_vmin", POINTER(c_double)),
        ("lc_v0", POINTER(c_double)),
        ("lc_J", POINTER(c_double)),
        ("lc_JP", POINTER(c_double)),
        ("lc_A", POINTER(c_double)),
        ("lc_f", POINTER(c_double)),
        ("lc_x", POINTER(c_double)),
        ("cache_x", POINTER(c_double)),
        ("scratch", POINTER(c_double)),
    ]

class MJMODEL(Structure):
    
    _fields_ = [
        ("nqpos", c_int),
        ("ndof", c_int),
        ("njnt", c_int),
        ("nbody", c_int),
        ("ngeom", c_int),
        ("neqmax", c_int),
        ("nlmax", c_int),
        ("ncpair", c_int),
        ("ncmax", c_int),
        ("nM", c_int),
        ("nbuffer", c_int),
        ("name", c_char * 100),
        ("timestep", c_double),
        ("gravity", c_double * 3),
        ("viscosity", c_double),
        ("mindist", c_double * 2),
        ("errreduce", c_double * 3),
        ("kinematic", c_ubyte),
        ("buffer", POINTER(c_ubyte)),
        ("qpos0", POINTER(c_double)),
        ("body_mass", POINTER(c_double)),
        ("body_inertia", POINTER(c_double)),
        ("body_pos", POINTER(c_double)),
        ("body_quat", POINTER(c_double)),
        ("body_viscoef", POINTER(c_double)),
        ("body_parent_id", POINTER(c_int)),
        ("body_jnt_num", POINTER(c_int)),
        ("body_jnt_adr", POINTER(c_int)),
        ("body_dof_num", POINTER(c_int)),
        ("body_dof_adr", POINTER(c_int)),
        ("body_root_id", POINTER(c_int)),
        ("jnt_type", POINTER(c_int)),
        ("jnt_pos", POINTER(c_double)),
        ("jnt_axis", POINTER(c_double)),
        ("jnt_spring", POINTER(c_double)),
        ("jnt_damping", POINTER(c_double)),
        ("jnt_vel_damping", POINTER(c_double)),
        ("jnt_qpos_adr", POINTER(c_int)),
        ("jnt_dof_adr", POINTER(c_int)),
        ("jnt_body_id", POINTER(c_int)),
        ("jnt_limit", POINTER(c_double)),
        ("jnt_islimited", POINTER(c_ubyte)),
        ("dof_armature", POINTER(c_double)),
        ("dof_body_id", POINTER(c_int)),
        ("dof_jnt_id", POINTER(c_int)),
        ("dof_parent_id", POINTER(c_int)),
        ("dof_M_adr", POINTER(c_int)),
        ("geom_type", POINTER(c_int)),
        ("geom_size", POINTER(c_double)),
        ("geom_pos", POINTER(c_double)),
        ("geom_quat", POINTER(c_double)),
        ("geom_color", POINTER(c_double)),
        ("geom_body_id", POINTER(c_int)),
        ("geom_isoffset", POINTER(c_ubyte)),
        ("eq_body1", POINTER(c_int)),
        ("eq_body2", POINTER(c_int)),
        ("eq_pos1", POINTER(c_double)),
        ("eq_pos2", POINTER(c_double)),
        ("eq_enable", POINTER(c_ubyte)),
        ("pair_type", POINTER(c_int)),
        ("pair_friction", POINTER(c_double)),
        ("pair_geom1", POINTER(c_int)),
        ("pair_geom2", POINTER(c_int)),
        ("pair_enable", POINTER(c_ubyte)),
    ]

class MjDataWrapper(object):
    
    def __init__(self, wrapped, size_src=None):
        self._wrapped = wrapped
        self._size_src = size_src

    @property
    def ptr(self):
        return self._wrapped

    @property
    def obj(self):
        return self._wrapped.contents

    
    @property
    def nbuffer(self):
        return self._wrapped.contents.nbuffer
    
    @nbuffer.setter
    def nbuffer(self, value):
        self._wrapped.contents.nbuffer = value
    
    @property
    def qpos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qpos, dtype=np.double, count=(self._size_src.nqpos*1)), (self._size_src.nqpos, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qpos.setter
    def qpos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qpos, val_ptr, self._size_src.nqpos*1 * sizeof(c_double))
    
    @property
    def qvel(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qvel, dtype=np.double, count=(self._size_src.ndof*1)), (self._size_src.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qvel.setter
    def qvel(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qvel, val_ptr, self._size_src.ndof*1 * sizeof(c_double))
    
    @property
    def qacc(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qacc, dtype=np.double, count=(self._size_src.ndof*1)), (self._size_src.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qacc.setter
    def qacc(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qacc, val_ptr, self._size_src.ndof*1 * sizeof(c_double))
    
    @property
    def qfrc_ext(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qfrc_ext, dtype=np.double, count=(self._size_src.ndof*1)), (self._size_src.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qfrc_ext.setter
    def qfrc_ext(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qfrc_ext, val_ptr, self._size_src.ndof*1 * sizeof(c_double))
    
    @property
    def qvel_next(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qvel_next, dtype=np.double, count=(self._size_src.ndof*1)), (self._size_src.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qvel_next.setter
    def qvel_next(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qvel_next, val_ptr, self._size_src.ndof*1 * sizeof(c_double))
    
    @property
    def xpos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.xpos, dtype=np.double, count=(self._size_src.nbody*3)), (self._size_src.nbody, 3, ))
        arr.setflags(write=False)
        return arr
    
    @xpos.setter
    def xpos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.xpos, val_ptr, self._size_src.nbody*3 * sizeof(c_double))
    
    @property
    def xquat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.xquat, dtype=np.double, count=(self._size_src.nbody*4)), (self._size_src.nbody, 4, ))
        arr.setflags(write=False)
        return arr
    
    @xquat.setter
    def xquat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.xquat, val_ptr, self._size_src.nbody*4 * sizeof(c_double))
    
    @property
    def xmat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.xmat, dtype=np.double, count=(self._size_src.nbody*9)), (self._size_src.nbody, 9, ))
        arr.setflags(write=False)
        return arr
    
    @xmat.setter
    def xmat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.xmat, val_ptr, self._size_src.nbody*9 * sizeof(c_double))
    
    @property
    def xanchor(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.xanchor, dtype=np.double, count=(self._size_src.njnt*3)), (self._size_src.njnt, 3, ))
        arr.setflags(write=False)
        return arr
    
    @xanchor.setter
    def xanchor(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.xanchor, val_ptr, self._size_src.njnt*3 * sizeof(c_double))
    
    @property
    def xaxis(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.xaxis, dtype=np.double, count=(self._size_src.njnt*3)), (self._size_src.njnt, 3, ))
        arr.setflags(write=False)
        return arr
    
    @xaxis.setter
    def xaxis(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.xaxis, val_ptr, self._size_src.njnt*3 * sizeof(c_double))
    
    @property
    def geom_xpos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_xpos, dtype=np.double, count=(self._size_src.ngeom*3)), (self._size_src.ngeom, 3, ))
        arr.setflags(write=False)
        return arr
    
    @geom_xpos.setter
    def geom_xpos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_xpos, val_ptr, self._size_src.ngeom*3 * sizeof(c_double))
    
    @property
    def geom_xmat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_xmat, dtype=np.double, count=(self._size_src.ngeom*9)), (self._size_src.ngeom, 9, ))
        arr.setflags(write=False)
        return arr
    
    @geom_xmat.setter
    def geom_xmat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_xmat, val_ptr, self._size_src.ngeom*9 * sizeof(c_double))
    
    @property
    def com(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.com, dtype=np.double, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @com.setter
    def com(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.com, val_ptr, 3 * sizeof(c_double))
    
    @property
    def cdof(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cdof, dtype=np.double, count=(self._size_src.ndof*6)), (self._size_src.ndof, 6, ))
        arr.setflags(write=False)
        return arr
    
    @cdof.setter
    def cdof(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cdof, val_ptr, self._size_src.ndof*6 * sizeof(c_double))
    
    @property
    def cinert(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cinert, dtype=np.double, count=(self._size_src.nbody*10)), (self._size_src.nbody, 10, ))
        arr.setflags(write=False)
        return arr
    
    @cinert.setter
    def cinert(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cinert, val_ptr, self._size_src.nbody*10 * sizeof(c_double))
    
    @property
    def cvel(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cvel, dtype=np.double, count=(self._size_src.nbody*6)), (self._size_src.nbody, 6, ))
        arr.setflags(write=False)
        return arr
    
    @cvel.setter
    def cvel(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cvel, val_ptr, self._size_src.nbody*6 * sizeof(c_double))
    
    @property
    def cacc(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cacc, dtype=np.double, count=(self._size_src.nbody*6)), (self._size_src.nbody, 6, ))
        arr.setflags(write=False)
        return arr
    
    @cacc.setter
    def cacc(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cacc, val_ptr, self._size_src.nbody*6 * sizeof(c_double))
    
    @property
    def cfrc_int(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cfrc_int, dtype=np.double, count=(self._size_src.nbody*6)), (self._size_src.nbody, 6, ))
        arr.setflags(write=False)
        return arr
    
    @cfrc_int.setter
    def cfrc_int(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cfrc_int, val_ptr, self._size_src.nbody*6 * sizeof(c_double))
    
    @property
    def qfrc_bias(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qfrc_bias, dtype=np.double, count=(self._size_src.ndof*1)), (self._size_src.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qfrc_bias.setter
    def qfrc_bias(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qfrc_bias, val_ptr, self._size_src.ndof*1 * sizeof(c_double))
    
    @property
    def crb(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.crb, dtype=np.double, count=(self._size_src.nbody*10)), (self._size_src.nbody, 10, ))
        arr.setflags(write=False)
        return arr
    
    @crb.setter
    def crb(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.crb, val_ptr, self._size_src.nbody*10 * sizeof(c_double))
    
    @property
    def qM(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qM, dtype=np.double, count=(self._size_src.nM*1)), (self._size_src.nM, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qM.setter
    def qM(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qM, val_ptr, self._size_src.nM*1 * sizeof(c_double))
    
    @property
    def qLD(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qLD, dtype=np.double, count=(self._size_src.nM*1)), (self._size_src.nM, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qLD.setter
    def qLD(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qLD, val_ptr, self._size_src.nM*1 * sizeof(c_double))
    
    @property
    def neq(self):
        return self._wrapped.contents.neq
    
    @neq.setter
    def neq(self, value):
        self._wrapped.contents.neq = value
    
    @property
    def eq_err(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_err, dtype=np.double, count=(3*self._size_src.neqmax*1)), (3*self._size_src.neqmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_err.setter
    def eq_err(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_err, val_ptr, 3*self._size_src.neqmax*1 * sizeof(c_double))
    
    @property
    def eq_v(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_v, dtype=np.double, count=(3*self._size_src.neqmax*1)), (3*self._size_src.neqmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_v.setter
    def eq_v(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_v, val_ptr, 3*self._size_src.neqmax*1 * sizeof(c_double))
    
    @property
    def eq_J(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_J, dtype=np.double, count=(3*self._size_src.neqmax*self._size_src.ndof)), (3*self._size_src.neqmax, self._size_src.ndof, ))
        arr.setflags(write=False)
        return arr
    
    @eq_J.setter
    def eq_J(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_J, val_ptr, 3*self._size_src.neqmax*self._size_src.ndof * sizeof(c_double))
    
    @property
    def eq_JMi(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_JMi, dtype=np.double, count=(3*self._size_src.neqmax*self._size_src.ndof)), (3*self._size_src.neqmax, self._size_src.ndof, ))
        arr.setflags(write=False)
        return arr
    
    @eq_JMi.setter
    def eq_JMi(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_JMi, val_ptr, 3*self._size_src.neqmax*self._size_src.ndof * sizeof(c_double))
    
    @property
    def nlim(self):
        return self._wrapped.contents.nlim
    
    @nlim.setter
    def nlim(self, value):
        self._wrapped.contents.nlim = value
    
    @property
    def ncon(self):
        return self._wrapped.contents.ncon
    
    @ncon.setter
    def ncon(self, value):
        self._wrapped.contents.ncon = value
    
    @property
    def lim_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lim_adr, dtype=np.int, count=(self._size_src.nlmax*1)), (self._size_src.nlmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @lim_adr.setter
    def lim_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.lim_adr, val_ptr, self._size_src.nlmax*1 * sizeof(c_int))
    
    @property
    def con_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.con_adr, dtype=np.int, count=(self._size_src.ncmax*1)), (self._size_src.ncmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @con_adr.setter
    def con_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.con_adr, val_ptr, self._size_src.ncmax*1 * sizeof(c_int))
    
    @property
    def con_pair(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.con_pair, dtype=np.int, count=(self._size_src.ncmax*1)), (self._size_src.ncmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @con_pair.setter
    def con_pair(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.con_pair, val_ptr, self._size_src.ncmax*1 * sizeof(c_int))
    
    @property
    def con_xpos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.con_xpos, dtype=np.double, count=(self._size_src.ncmax*3)), (self._size_src.ncmax, 3, ))
        arr.setflags(write=False)
        return arr
    
    @con_xpos.setter
    def con_xpos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.con_xpos, val_ptr, self._size_src.ncmax*3 * sizeof(c_double))
    
    @property
    def con_xmatT(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.con_xmatT, dtype=np.double, count=(self._size_src.ncmax*9)), (self._size_src.ncmax, 9, ))
        arr.setflags(write=False)
        return arr
    
    @con_xmatT.setter
    def con_xmatT(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.con_xmatT, val_ptr, self._size_src.ncmax*9 * sizeof(c_double))
    
    @property
    def con_friction(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.con_friction, dtype=np.double, count=(self._size_src.ncmax*1)), (self._size_src.ncmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @con_friction.setter
    def con_friction(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.con_friction, val_ptr, self._size_src.ncmax*1 * sizeof(c_double))
    
    @property
    def lc_v0(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lc_v0, dtype=np.double, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @lc_v0.setter
    def lc_v0(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.lc_v0, val_ptr, self._size_src.njmax*1 * sizeof(c_double))
    
    @property
    def lc_J(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lc_J, dtype=np.double, count=(self._size_src.njmax*self._size_src.ndof)), (self._size_src.njmax, self._size_src.ndof, ))
        arr.setflags(write=False)
        return arr
    
    @lc_J.setter
    def lc_J(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.lc_J, val_ptr, self._size_src.njmax*self._size_src.ndof * sizeof(c_double))
    
    @property
    def lc_JP(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lc_JP, dtype=np.double, count=(self._size_src.njmax*self._size_src.ndof)), (self._size_src.njmax, self._size_src.ndof, ))
        arr.setflags(write=False)
        return arr
    
    @lc_JP.setter
    def lc_JP(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.lc_JP, val_ptr, self._size_src.njmax*self._size_src.ndof * sizeof(c_double))
    
    @property
    def lc_A(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lc_A, dtype=np.double, count=(self._size_src.njmax*self._size_src.njmax)), (self._size_src.njmax, self._size_src.njmax, ))
        arr.setflags(write=False)
        return arr
    
    @lc_A.setter
    def lc_A(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.lc_A, val_ptr, self._size_src.njmax*self._size_src.njmax * sizeof(c_double))
    
    @property
    def lc_f(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lc_f, dtype=np.double, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @lc_f.setter
    def lc_f(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.lc_f, val_ptr, self._size_src.njmax*1 * sizeof(c_double))
    
    @property
    def lc_x(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lc_x, dtype=np.double, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @lc_x.setter
    def lc_x(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.lc_x, val_ptr, self._size_src.njmax*1 * sizeof(c_double))
    
    @property
    def cache_x(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cache_x, dtype=np.double, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @cache_x.setter
    def cache_x(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cache_x, val_ptr, self._size_src.njmax*1 * sizeof(c_double))

class MjModelWrapper(object):
    
    def __init__(self, wrapped, size_src=None):
        self._wrapped = wrapped
        self._size_src = size_src

    @property
    def ptr(self):
        return self._wrapped

    @property
    def obj(self):
        return self._wrapped.contents

    
    @property
    def nqpos(self):
        return self._wrapped.contents.nqpos
    
    @nqpos.setter
    def nqpos(self, value):
        self._wrapped.contents.nqpos = value
    
    @property
    def ndof(self):
        return self._wrapped.contents.ndof
    
    @ndof.setter
    def ndof(self, value):
        self._wrapped.contents.ndof = value
    
    @property
    def njnt(self):
        return self._wrapped.contents.njnt
    
    @njnt.setter
    def njnt(self, value):
        self._wrapped.contents.njnt = value
    
    @property
    def nbody(self):
        return self._wrapped.contents.nbody
    
    @nbody.setter
    def nbody(self, value):
        self._wrapped.contents.nbody = value
    
    @property
    def ngeom(self):
        return self._wrapped.contents.ngeom
    
    @ngeom.setter
    def ngeom(self, value):
        self._wrapped.contents.ngeom = value
    
    @property
    def neqmax(self):
        return self._wrapped.contents.neqmax
    
    @neqmax.setter
    def neqmax(self, value):
        self._wrapped.contents.neqmax = value
    
    @property
    def nlmax(self):
        return self._wrapped.contents.nlmax
    
    @nlmax.setter
    def nlmax(self, value):
        self._wrapped.contents.nlmax = value
    
    @property
    def ncpair(self):
        return self._wrapped.contents.ncpair
    
    @ncpair.setter
    def ncpair(self, value):
        self._wrapped.contents.ncpair = value
    
    @property
    def ncmax(self):
        return self._wrapped.contents.ncmax
    
    @ncmax.setter
    def ncmax(self, value):
        self._wrapped.contents.ncmax = value
    
    @property
    def nM(self):
        return self._wrapped.contents.nM
    
    @nM.setter
    def nM(self, value):
        self._wrapped.contents.nM = value
    
    @property
    def nbuffer(self):
        return self._wrapped.contents.nbuffer
    
    @nbuffer.setter
    def nbuffer(self, value):
        self._wrapped.contents.nbuffer = value
    
    @property
    def name(self):
        return self._wrapped.contents.name
    
    @name.setter
    def name(self, value):
        self._wrapped.contents.name = value
    
    @property
    def timestep(self):
        return self._wrapped.contents.timestep
    
    @timestep.setter
    def timestep(self, value):
        self._wrapped.contents.timestep = value
    
    @property
    def gravity(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.gravity, dtype=np.double, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @gravity.setter
    def gravity(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.gravity, val_ptr, 3 * sizeof(c_double))
    
    @property
    def viscosity(self):
        return self._wrapped.contents.viscosity
    
    @viscosity.setter
    def viscosity(self, value):
        self._wrapped.contents.viscosity = value
    
    @property
    def mindist(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mindist, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @mindist.setter
    def mindist(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.mindist, val_ptr, 2 * sizeof(c_double))
    
    @property
    def errreduce(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.errreduce, dtype=np.double, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @errreduce.setter
    def errreduce(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.errreduce, val_ptr, 3 * sizeof(c_double))
    
    @property
    def kinematic(self):
        return self._wrapped.contents.kinematic
    
    @kinematic.setter
    def kinematic(self, value):
        self._wrapped.contents.kinematic = value
    
    @property
    def qpos0(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qpos0, dtype=np.double, count=(self.nqpos*1)), (self.nqpos, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qpos0.setter
    def qpos0(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qpos0, val_ptr, self.nqpos*1 * sizeof(c_double))
    
    @property
    def body_mass(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_mass, dtype=np.double, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_mass.setter
    def body_mass(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_mass, val_ptr, self.nbody*1 * sizeof(c_double))
    
    @property
    def body_inertia(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_inertia, dtype=np.double, count=(self.nbody*3)), (self.nbody, 3, ))
        arr.setflags(write=False)
        return arr
    
    @body_inertia.setter
    def body_inertia(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_inertia, val_ptr, self.nbody*3 * sizeof(c_double))
    
    @property
    def body_pos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_pos, dtype=np.double, count=(self.nbody*3)), (self.nbody, 3, ))
        arr.setflags(write=False)
        return arr
    
    @body_pos.setter
    def body_pos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_pos, val_ptr, self.nbody*3 * sizeof(c_double))
    
    @property
    def body_quat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_quat, dtype=np.double, count=(self.nbody*4)), (self.nbody, 4, ))
        arr.setflags(write=False)
        return arr
    
    @body_quat.setter
    def body_quat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_quat, val_ptr, self.nbody*4 * sizeof(c_double))
    
    @property
    def body_viscoef(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_viscoef, dtype=np.double, count=(self.nbody*6)), (self.nbody, 6, ))
        arr.setflags(write=False)
        return arr
    
    @body_viscoef.setter
    def body_viscoef(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_viscoef, val_ptr, self.nbody*6 * sizeof(c_double))
    
    @property
    def body_parent_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_parent_id, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_parent_id.setter
    def body_parent_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_parent_id, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_jnt_num(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_jnt_num, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_jnt_num.setter
    def body_jnt_num(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_jnt_num, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_jnt_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_jnt_adr, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_jnt_adr.setter
    def body_jnt_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_jnt_adr, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_dof_num(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_dof_num, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_dof_num.setter
    def body_dof_num(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_dof_num, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_dof_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_dof_adr, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_dof_adr.setter
    def body_dof_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_dof_adr, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_root_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_root_id, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_root_id.setter
    def body_root_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_root_id, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def jnt_pos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_pos, dtype=np.double, count=(self.njnt*3)), (self.njnt, 3, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_pos.setter
    def jnt_pos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_pos, val_ptr, self.njnt*3 * sizeof(c_double))
    
    @property
    def jnt_axis(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_axis, dtype=np.double, count=(self.njnt*3)), (self.njnt, 3, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_axis.setter
    def jnt_axis(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_axis, val_ptr, self.njnt*3 * sizeof(c_double))
    
    @property
    def jnt_spring(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_spring, dtype=np.double, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_spring.setter
    def jnt_spring(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_spring, val_ptr, self.njnt*1 * sizeof(c_double))
    
    @property
    def jnt_damping(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_damping, dtype=np.double, count=(self.njnt*2)), (self.njnt, 2, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_damping.setter
    def jnt_damping(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_damping, val_ptr, self.njnt*2 * sizeof(c_double))
    
    @property
    def jnt_vel_damping(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_vel_damping, dtype=np.double, count=(self.njnt*2)), (self.njnt, 2, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_vel_damping.setter
    def jnt_vel_damping(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_vel_damping, val_ptr, self.njnt*2 * sizeof(c_double))
    
    @property
    def jnt_qpos_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_qpos_adr, dtype=np.int, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_qpos_adr.setter
    def jnt_qpos_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.jnt_qpos_adr, val_ptr, self.njnt*1 * sizeof(c_int))
    
    @property
    def jnt_dof_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_dof_adr, dtype=np.int, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_dof_adr.setter
    def jnt_dof_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.jnt_dof_adr, val_ptr, self.njnt*1 * sizeof(c_int))
    
    @property
    def jnt_body_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_body_id, dtype=np.int, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_body_id.setter
    def jnt_body_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.jnt_body_id, val_ptr, self.njnt*1 * sizeof(c_int))
    
    @property
    def jnt_limit(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_limit, dtype=np.double, count=(self.njnt*2)), (self.njnt, 2, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_limit.setter
    def jnt_limit(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_limit, val_ptr, self.njnt*2 * sizeof(c_double))
    
    @property
    def jnt_islimited(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_islimited, dtype=np.uint8, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_islimited.setter
    def jnt_islimited(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.jnt_islimited, val_ptr, self.njnt*1 * sizeof(c_ubyte))
    
    @property
    def dof_armature(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_armature, dtype=np.double, count=(self.ndof*1)), (self.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_armature.setter
    def dof_armature(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.dof_armature, val_ptr, self.ndof*1 * sizeof(c_double))
    
    @property
    def dof_body_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_body_id, dtype=np.int, count=(self.ndof*1)), (self.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_body_id.setter
    def dof_body_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.dof_body_id, val_ptr, self.ndof*1 * sizeof(c_int))
    
    @property
    def dof_jnt_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_jnt_id, dtype=np.int, count=(self.ndof*1)), (self.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_jnt_id.setter
    def dof_jnt_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.dof_jnt_id, val_ptr, self.ndof*1 * sizeof(c_int))
    
    @property
    def dof_parent_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_parent_id, dtype=np.int, count=(self.ndof*1)), (self.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_parent_id.setter
    def dof_parent_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.dof_parent_id, val_ptr, self.ndof*1 * sizeof(c_int))
    
    @property
    def dof_M_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_M_adr, dtype=np.int, count=(self.ndof*1)), (self.ndof, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_M_adr.setter
    def dof_M_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.dof_M_adr, val_ptr, self.ndof*1 * sizeof(c_int))
    
    @property
    def geom_size(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_size, dtype=np.double, count=(self.ngeom*self._size_src.mjNumSize)), (self.ngeom, self._size_src.mjNumSize, ))
        arr.setflags(write=False)
        return arr
    
    @geom_size.setter
    def geom_size(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_size, val_ptr, self.ngeom*self._size_src.mjNumSize * sizeof(c_double))
    
    @property
    def geom_pos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_pos, dtype=np.double, count=(self.ngeom*3)), (self.ngeom, 3, ))
        arr.setflags(write=False)
        return arr
    
    @geom_pos.setter
    def geom_pos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_pos, val_ptr, self.ngeom*3 * sizeof(c_double))
    
    @property
    def geom_quat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_quat, dtype=np.double, count=(self.ngeom*4)), (self.ngeom, 4, ))
        arr.setflags(write=False)
        return arr
    
    @geom_quat.setter
    def geom_quat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_quat, val_ptr, self.ngeom*4 * sizeof(c_double))
    
    @property
    def geom_color(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_color, dtype=np.double, count=(self.ngeom*3)), (self.ngeom, 3, ))
        arr.setflags(write=False)
        return arr
    
    @geom_color.setter
    def geom_color(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_color, val_ptr, self.ngeom*3 * sizeof(c_double))
    
    @property
    def geom_body_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_body_id, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_body_id.setter
    def geom_body_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.geom_body_id, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def geom_isoffset(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_isoffset, dtype=np.uint8, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_isoffset.setter
    def geom_isoffset(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.geom_isoffset, val_ptr, self.ngeom*1 * sizeof(c_ubyte))
    
    @property
    def eq_body1(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_body1, dtype=np.int, count=(self.neqmax*1)), (self.neqmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_body1.setter
    def eq_body1(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_body1, val_ptr, self.neqmax*1 * sizeof(c_int))
    
    @property
    def eq_body2(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_body2, dtype=np.int, count=(self.neqmax*1)), (self.neqmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_body2.setter
    def eq_body2(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_body2, val_ptr, self.neqmax*1 * sizeof(c_int))
    
    @property
    def eq_pos1(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_pos1, dtype=np.double, count=(self.neqmax*3)), (self.neqmax, 3, ))
        arr.setflags(write=False)
        return arr
    
    @eq_pos1.setter
    def eq_pos1(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_pos1, val_ptr, self.neqmax*3 * sizeof(c_double))
    
    @property
    def eq_pos2(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_pos2, dtype=np.double, count=(self.neqmax*3)), (self.neqmax, 3, ))
        arr.setflags(write=False)
        return arr
    
    @eq_pos2.setter
    def eq_pos2(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_pos2, val_ptr, self.neqmax*3 * sizeof(c_double))
    
    @property
    def eq_enable(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_enable, dtype=np.uint8, count=(self.neqmax*1)), (self.neqmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_enable.setter
    def eq_enable(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.eq_enable, val_ptr, self.neqmax*1 * sizeof(c_ubyte))
    
    @property
    def pair_friction(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_friction, dtype=np.double, count=(self.ncpair*1)), (self.ncpair, 1, ))
        arr.setflags(write=False)
        return arr
    
    @pair_friction.setter
    def pair_friction(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.pair_friction, val_ptr, self.ncpair*1 * sizeof(c_double))
    
    @property
    def pair_geom1(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_geom1, dtype=np.int, count=(self.ncpair*1)), (self.ncpair, 1, ))
        arr.setflags(write=False)
        return arr
    
    @pair_geom1.setter
    def pair_geom1(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.pair_geom1, val_ptr, self.ncpair*1 * sizeof(c_int))
    
    @property
    def pair_geom2(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_geom2, dtype=np.int, count=(self.ncpair*1)), (self.ncpair, 1, ))
        arr.setflags(write=False)
        return arr
    
    @pair_geom2.setter
    def pair_geom2(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.pair_geom2, val_ptr, self.ncpair*1 * sizeof(c_int))
    
    @property
    def pair_enable(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_enable, dtype=np.uint8, count=(self.ncpair*1)), (self.ncpair, 1, ))
        arr.setflags(write=False)
        return arr
    
    @pair_enable.setter
    def pair_enable(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.pair_enable, val_ptr, self.ncpair*1 * sizeof(c_ubyte))
