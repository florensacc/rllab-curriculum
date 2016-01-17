
# AUTO GENERATED. DO NOT CHANGE!
from ctypes import *
import numpy as np

class MJVROPE(Structure):
    
    _fields_ = [
        ("cursor", c_double * 3),
        ("endpoint", c_double * 3),
        ("length", c_double),
        ("distplane", c_double),
        ("sag", c_double),
    ]

class MJVCAMERA(Structure):
    
    _fields_ = [
        ("fov", c_float * 2),
        ("dir", c_float * 3),
        ("up", c_float * 3),
        ("pos", c_float * 3),
        ("lookat", c_float * 3),
        ("azimuth", c_float),
        ("elevation", c_float),
        ("distance", c_float),
        ("ipd", c_float),
        ("fixcam", c_int),
        ("trackbody", c_int),
        ("rope", MJVROPE),
    ]

class MJVGEOM(Structure):
    
    _fields_ = [
        ("type", c_int),
        ("dataid", c_int),
        ("objtype", c_int),
        ("objid", c_int),
        ("category", c_int),
        ("flags", c_int),
        ("size", c_float * 3),
        ("pos", c_float * 3),
        ("mat", c_float * 9),
        ("rgba", c_float * 4),
        ("label", c_char * 40),
        ("camdist", c_float),
        ("rbound", c_float),
        ("transparent", c_ubyte),
    ]

class MJOPTION(Structure):
    
    _fields_ = [
        ("timestep", c_double),
        ("gravity", c_double * 3),
        ("wind", c_double * 3),
        ("density", c_double),
        ("viscosity", c_double),
        ("expdist", c_double),
        ("s_mindist", c_double * 2),
        ("s_stiffness", c_double * 2),
        ("s_damping", c_double * 2),
        ("s_armature", c_double * 2),
        ("s_friction", c_double * 2),
        ("s_frictionloss", c_double * 2),
        ("s_compliance", c_double * 2),
        ("s_timeconst", c_double * 2),
        ("disableflags", c_int),
        ("integrator", c_int),
        ("collisionmode", c_int),
        ("algorithm", c_int),
        ("iterations", c_int),
        ("eqsoft", c_ubyte),
        ("fastdiag", c_ubyte),
        ("remotecontact", c_ubyte),
        ("stats", c_ubyte),
    ]

class MJDATA(Structure):
    
    _fields_ = [
        ("nstack", c_int),
        ("nbuffer", c_int),
        ("pstack", c_int),
        ("maxstackuse", c_int),
        ("ne", c_int),
        ("nc", c_int),
        ("nfri", c_int),
        ("nlim", c_int),
        ("ncon", c_int),
        ("nflc", c_int),
        ("nstat", c_int),
        ("nwarning", c_int * 7),
        ("timer", (c_double * 12) * 2),
        ("stat", (c_double * 100) * 2),
        ("time", c_double),
        ("energy", c_double * 2),
        ("buffer", POINTER(c_ubyte)),
        ("stack", POINTER(c_double)),
        ("userdata", POINTER(c_double)),
        ("qpos", POINTER(c_double)),
        ("qvel", POINTER(c_double)),
        ("qvel_next", POINTER(c_double)),
        ("act", POINTER(c_double)),
        ("act_next", POINTER(c_double)),
        ("ctrl", POINTER(c_double)),
        ("qfrc_applied", POINTER(c_double)),
        ("xfrc_applied", POINTER(c_double)),
        ("qfrc_bias", POINTER(c_double)),
        ("qfrc_passive", POINTER(c_double)),
        ("qfrc_actuation", POINTER(c_double)),
        ("qfrc_impulse", POINTER(c_double)),
        ("qfrc_constraint", POINTER(c_double)),
        ("xpos", POINTER(c_double)),
        ("xquat", POINTER(c_double)),
        ("xmat", POINTER(c_double)),
        ("xipos", POINTER(c_double)),
        ("ximat", POINTER(c_double)),
        ("xanchor", POINTER(c_double)),
        ("xaxis", POINTER(c_double)),
        ("geom_xpos", POINTER(c_double)),
        ("geom_xmat", POINTER(c_double)),
        ("site_xpos", POINTER(c_double)),
        ("site_xmat", POINTER(c_double)),
        ("com_subtree", POINTER(c_double)),
        ("cdof", POINTER(c_double)),
        ("cinert", POINTER(c_double)),
        ("ten_wrapadr", POINTER(c_int)),
        ("ten_wrapnum", POINTER(c_int)),
        ("ten_length", POINTER(c_double)),
        ("ten_moment", POINTER(c_double)),
        ("wrap_obj", POINTER(c_int)),
        ("wrap_xpos", POINTER(c_double)),
        ("actuator_length", POINTER(c_double)),
        ("actuator_moment", POINTER(c_double)),
        ("actuator_force", POINTER(c_double)),
        ("cvel", POINTER(c_double)),
        ("cacc", POINTER(c_double)),
        ("cfrc_int", POINTER(c_double)),
        ("cfrc_ext", POINTER(c_double)),
        ("qM", POINTER(c_double)),
        ("qD", POINTER(c_double)),
        ("qLD", POINTER(c_double)),
        ("qLDiagSqr", POINTER(c_double)),
        ("eq_id", POINTER(c_int)),
        ("eq_err", POINTER(c_double)),
        ("eq_J", POINTER(c_double)),
        ("eq_vdes", POINTER(c_double)),
        ("eq_JMi", POINTER(c_double)),
        ("eq_Achol", POINTER(c_double)),
        ("eq_R", POINTER(c_double)),
        ("contact", POINTER(c_void_p)),
        ("fri_id", POINTER(c_int)),
        ("lim_id", POINTER(c_int)),
        ("con_id", POINTER(c_int)),
        ("lc_ind", POINTER(c_int)),
        ("flc_signature", POINTER(c_int)),
        ("lc_dist", POINTER(c_double)),
        ("flc_J", POINTER(c_double)),
        ("flc_ek", POINTER(c_double)),
        ("flc_A", POINTER(c_double)),
        ("flc_R", POINTER(c_double)),
        ("flc_vdes", POINTER(c_double)),
        ("flc_b", POINTER(c_double)),
        ("flc_f", POINTER(c_double)),
    ]

class MJMODEL(Structure):
    
    _fields_ = [
        ("nq", c_int),
        ("nv", c_int),
        ("nu", c_int),
        ("na", c_int),
        ("nbody", c_int),
        ("njnt", c_int),
        ("ngeom", c_int),
        ("nsite", c_int),
        ("ncam", c_int),
        ("nmesh", c_int),
        ("nmeshvert", c_int),
        ("nmeshface", c_int),
        ("nmeshgraph", c_int),
        ("nhfield", c_int),
        ("nhfielddata", c_int),
        ("ncpair", c_int),
        ("neq", c_int),
        ("ntendon", c_int),
        ("nwrap", c_int),
        ("nnumeric", c_int),
        ("nnumericdata", c_int),
        ("ntext", c_int),
        ("ntextdata", c_int),
        ("nuser_body", c_int),
        ("nuser_jnt", c_int),
        ("nuser_geom", c_int),
        ("nuser_site", c_int),
        ("nuser_eq", c_int),
        ("nuser_tendon", c_int),
        ("nuser_actuator", c_int),
        ("nnames", c_int),
        ("nM", c_int),
        ("nemax", c_int),
        ("nlmax", c_int),
        ("ncmax", c_int),
        ("njmax", c_int),
        ("nctotmax", c_int),
        ("nstack", c_int),
        ("nuserdata", c_int),
        ("nbuffer", c_int),
        ("buffer", POINTER(c_ubyte)),
        ("qpos0", POINTER(c_double)),
        ("qpos_spring", POINTER(c_double)),
        ("body_parentid", POINTER(c_int)),
        ("body_rootid", POINTER(c_int)),
        ("body_weldid", POINTER(c_int)),
        ("body_jntnum", POINTER(c_int)),
        ("body_jntadr", POINTER(c_int)),
        ("body_dofnum", POINTER(c_int)),
        ("body_dofadr", POINTER(c_int)),
        ("body_geomnum", POINTER(c_int)),
        ("body_geomadr", POINTER(c_int)),
        ("body_gravcomp", POINTER(c_ubyte)),
        ("body_pos", POINTER(c_double)),
        ("body_quat", POINTER(c_double)),
        ("body_ipos", POINTER(c_double)),
        ("body_iquat", POINTER(c_double)),
        ("body_mass", POINTER(c_double)),
        ("body_inertia", POINTER(c_double)),
        ("body_invweight0", POINTER(c_double)),
        ("body_user", POINTER(c_double)),
        ("jnt_type", POINTER(c_int)),
        ("jnt_qposadr", POINTER(c_int)),
        ("jnt_dofadr", POINTER(c_int)),
        ("jnt_bodyid", POINTER(c_int)),
        ("jnt_islimited", POINTER(c_ubyte)),
        ("jnt_pos", POINTER(c_double)),
        ("jnt_axis", POINTER(c_double)),
        ("jnt_stiffness", POINTER(c_double)),
        ("jnt_range", POINTER(c_double)),
        ("jnt_compliance", POINTER(c_double)),
        ("jnt_timeconst", POINTER(c_double)),
        ("jnt_mindist", POINTER(c_double)),
        ("jnt_user", POINTER(c_double)),
        ("dof_bodyid", POINTER(c_int)),
        ("dof_jntid", POINTER(c_int)),
        ("dof_parentid", POINTER(c_int)),
        ("dof_Madr", POINTER(c_int)),
        ("dof_isfrictional", POINTER(c_ubyte)),
        ("dof_armature", POINTER(c_double)),
        ("dof_damping", POINTER(c_double)),
        ("dof_frictionloss", POINTER(c_double)),
        ("dof_maxvel", POINTER(c_double)),
        ("dof_invweight0", POINTER(c_double)),
        ("geom_type", POINTER(c_int)),
        ("geom_contype", POINTER(c_int)),
        ("geom_conaffinity", POINTER(c_int)),
        ("geom_condim", POINTER(c_int)),
        ("geom_bodyid", POINTER(c_int)),
        ("geom_dataid", POINTER(c_int)),
        ("geom_group", POINTER(c_int)),
        ("geom_size", POINTER(c_double)),
        ("geom_rbound", POINTER(c_double)),
        ("geom_pos", POINTER(c_double)),
        ("geom_quat", POINTER(c_double)),
        ("geom_friction", POINTER(c_double)),
        ("geom_compliance", POINTER(c_double)),
        ("geom_timeconst", POINTER(c_double)),
        ("geom_mindist", POINTER(c_double)),
        ("geom_user", POINTER(c_double)),
        ("geom_rgba", POINTER(c_float)),
        ("site_bodyid", POINTER(c_int)),
        ("site_group", POINTER(c_int)),
        ("site_pos", POINTER(c_double)),
        ("site_quat", POINTER(c_double)),
        ("site_user", POINTER(c_double)),
        ("mesh_faceadr", POINTER(c_int)),
        ("mesh_facenum", POINTER(c_int)),
        ("mesh_vertadr", POINTER(c_int)),
        ("mesh_vertnum", POINTER(c_int)),
        ("mesh_graphadr", POINTER(c_int)),
        ("mesh_vert", POINTER(c_float)),
        ("mesh_face", POINTER(c_int)),
        ("mesh_graph", POINTER(c_int)),
        ("hfield_nrow", POINTER(c_int)),
        ("hfield_ncol", POINTER(c_int)),
        ("hfield_adr", POINTER(c_int)),
        ("hfield_geomid", POINTER(c_int)),
        ("hfield_dir", POINTER(c_ubyte)),
        ("hfield_data", POINTER(c_float)),
        ("pair_dim", POINTER(c_int)),
        ("pair_geom1", POINTER(c_int)),
        ("pair_geom2", POINTER(c_int)),
        ("pair_compliance", POINTER(c_double)),
        ("pair_timeconst", POINTER(c_double)),
        ("pair_mindist", POINTER(c_double)),
        ("pair_friction", POINTER(c_double)),
        ("eq_type", POINTER(c_int)),
        ("eq_obj1type", POINTER(c_int)),
        ("eq_obj2type", POINTER(c_int)),
        ("eq_obj1id", POINTER(c_int)),
        ("eq_obj2id", POINTER(c_int)),
        ("eq_size", POINTER(c_int)),
        ("eq_ndata", POINTER(c_int)),
        ("eq_isactive", POINTER(c_ubyte)),
        ("eq_compliance", POINTER(c_double)),
        ("eq_timeconst", POINTER(c_double)),
        ("eq_data", POINTER(c_double)),
        ("eq_user", POINTER(c_double)),
        ("tendon_adr", POINTER(c_int)),
        ("tendon_num", POINTER(c_int)),
        ("tendon_islimited", POINTER(c_ubyte)),
        ("tendon_compliance", POINTER(c_double)),
        ("tendon_timeconst", POINTER(c_double)),
        ("tendon_range", POINTER(c_double)),
        ("tendon_mindist", POINTER(c_double)),
        ("tendon_stiffness", POINTER(c_double)),
        ("tendon_damping", POINTER(c_double)),
        ("tendon_lengthspring", POINTER(c_double)),
        ("tendon_invweight0", POINTER(c_double)),
        ("tendon_user", POINTER(c_double)),
        ("wrap_type", POINTER(c_int)),
        ("wrap_objid", POINTER(c_int)),
        ("wrap_prm", POINTER(c_double)),
        ("actuator_dyntype", POINTER(c_int)),
        ("actuator_trntype", POINTER(c_int)),
        ("actuator_gaintype", POINTER(c_int)),
        ("actuator_biastype", POINTER(c_int)),
        ("actuator_trnid", POINTER(c_int)),
        ("actuator_isctrllimited", POINTER(c_ubyte)),
        ("actuator_isforcelimited", POINTER(c_ubyte)),
        ("actuator_dynprm", POINTER(c_double)),
        ("actuator_trnprm", POINTER(c_double)),
        ("actuator_gainprm", POINTER(c_double)),
        ("actuator_biasprm", POINTER(c_double)),
        ("actuator_ctrlrange", POINTER(c_double)),
        ("actuator_forcerange", POINTER(c_double)),
        ("actuator_invweight0", POINTER(c_double)),
        ("actuator_length0", POINTER(c_double)),
        ("actuator_lengthrange", POINTER(c_double)),
        ("actuator_user", POINTER(c_double)),
        ("numeric_adr", POINTER(c_int)),
        ("numeric_size", POINTER(c_int)),
        ("numeric_data", POINTER(c_double)),
        ("text_adr", POINTER(c_int)),
        ("text_data", POINTER(c_char)),
        ("name_bodyadr", POINTER(c_int)),
        ("name_jntadr", POINTER(c_int)),
        ("name_geomadr", POINTER(c_int)),
        ("name_siteadr", POINTER(c_int)),
        ("name_meshadr", POINTER(c_int)),
        ("name_hfieldadr", POINTER(c_int)),
        ("name_eqadr", POINTER(c_int)),
        ("name_tendonadr", POINTER(c_int)),
        ("name_actuatoradr", POINTER(c_int)),
        ("name_numericadr", POINTER(c_int)),
        ("name_textadr", POINTER(c_int)),
        ("names", POINTER(c_char)),
        ("key_qpos", POINTER(c_double)),
        ("key_qvel", POINTER(c_double)),
        ("key_act", POINTER(c_double)),
        ("cam_objtype", POINTER(c_int)),
        ("cam_objid", POINTER(c_int)),
        ("cam_resolution", POINTER(c_int)),
        ("cam_fov", POINTER(c_double)),
        ("cam_ipd", POINTER(c_double)),
    ]

class MjvRopeWrapper(object):
    
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
    def cursor(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cursor, dtype=np.double, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @cursor.setter
    def cursor(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cursor, val_ptr, 3 * sizeof(c_double))
    
    @property
    def endpoint(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.endpoint, dtype=np.double, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @endpoint.setter
    def endpoint(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.endpoint, val_ptr, 3 * sizeof(c_double))
    
    @property
    def length(self):
        return self._wrapped.contents.length
    
    @length.setter
    def length(self, value):
        self._wrapped.contents.length = value
    
    @property
    def distplane(self):
        return self._wrapped.contents.distplane
    
    @distplane.setter
    def distplane(self, value):
        self._wrapped.contents.distplane = value
    
    @property
    def sag(self):
        return self._wrapped.contents.sag
    
    @sag.setter
    def sag(self, value):
        self._wrapped.contents.sag = value

class MjvCameraWrapper(object):
    
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
    def fov(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.fov, dtype=np.float, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @fov.setter
    def fov(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.fov, val_ptr, 2 * sizeof(c_float))
    
    @property
    def dir(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dir, dtype=np.float, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @dir.setter
    def dir(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.dir, val_ptr, 3 * sizeof(c_float))
    
    @property
    def up(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.up, dtype=np.float, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @up.setter
    def up(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.up, val_ptr, 3 * sizeof(c_float))
    
    @property
    def pos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pos, dtype=np.float, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @pos.setter
    def pos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.pos, val_ptr, 3 * sizeof(c_float))
    
    @property
    def lookat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lookat, dtype=np.float, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @lookat.setter
    def lookat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.lookat, val_ptr, 3 * sizeof(c_float))
    
    @property
    def azimuth(self):
        return self._wrapped.contents.azimuth
    
    @azimuth.setter
    def azimuth(self, value):
        self._wrapped.contents.azimuth = value
    
    @property
    def elevation(self):
        return self._wrapped.contents.elevation
    
    @elevation.setter
    def elevation(self, value):
        self._wrapped.contents.elevation = value
    
    @property
    def distance(self):
        return self._wrapped.contents.distance
    
    @distance.setter
    def distance(self, value):
        self._wrapped.contents.distance = value
    
    @property
    def ipd(self):
        return self._wrapped.contents.ipd
    
    @ipd.setter
    def ipd(self, value):
        self._wrapped.contents.ipd = value
    
    @property
    def fixcam(self):
        return self._wrapped.contents.fixcam
    
    @fixcam.setter
    def fixcam(self, value):
        self._wrapped.contents.fixcam = value
    
    @property
    def trackbody(self):
        return self._wrapped.contents.trackbody
    
    @trackbody.setter
    def trackbody(self, value):
        self._wrapped.contents.trackbody = value
    
    @property
    def rope(self):
        return self._wrapped.contents.rope
    
    @rope.setter
    def rope(self, value):
        self._wrapped.contents.rope = value

class MjvGeomWrapper(object):
    
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
    def type(self):
        return self._wrapped.contents.type
    
    @type.setter
    def type(self, value):
        self._wrapped.contents.type = value
    
    @property
    def dataid(self):
        return self._wrapped.contents.dataid
    
    @dataid.setter
    def dataid(self, value):
        self._wrapped.contents.dataid = value
    
    @property
    def objtype(self):
        return self._wrapped.contents.objtype
    
    @objtype.setter
    def objtype(self, value):
        self._wrapped.contents.objtype = value
    
    @property
    def objid(self):
        return self._wrapped.contents.objid
    
    @objid.setter
    def objid(self, value):
        self._wrapped.contents.objid = value
    
    @property
    def category(self):
        return self._wrapped.contents.category
    
    @category.setter
    def category(self, value):
        self._wrapped.contents.category = value
    
    @property
    def flags(self):
        return self._wrapped.contents.flags
    
    @flags.setter
    def flags(self, value):
        self._wrapped.contents.flags = value
    
    @property
    def size(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.size, dtype=np.float, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @size.setter
    def size(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.size, val_ptr, 3 * sizeof(c_float))
    
    @property
    def pos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pos, dtype=np.float, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @pos.setter
    def pos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.pos, val_ptr, 3 * sizeof(c_float))
    
    @property
    def mat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mat, dtype=np.float, count=(9)), (9, ))
        arr.setflags(write=False)
        return arr
    
    @mat.setter
    def mat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.mat, val_ptr, 9 * sizeof(c_float))
    
    @property
    def rgba(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.rgba, dtype=np.float, count=(4)), (4, ))
        arr.setflags(write=False)
        return arr
    
    @rgba.setter
    def rgba(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.rgba, val_ptr, 4 * sizeof(c_float))
    
    @property
    def label(self):
        return self._wrapped.contents.label
    
    @label.setter
    def label(self, value):
        self._wrapped.contents.label = value
    
    @property
    def camdist(self):
        return self._wrapped.contents.camdist
    
    @camdist.setter
    def camdist(self, value):
        self._wrapped.contents.camdist = value
    
    @property
    def rbound(self):
        return self._wrapped.contents.rbound
    
    @rbound.setter
    def rbound(self, value):
        self._wrapped.contents.rbound = value
    
    @property
    def transparent(self):
        return self._wrapped.contents.transparent
    
    @transparent.setter
    def transparent(self, value):
        self._wrapped.contents.transparent = value

class MjOptionWrapper(object):
    
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
    def wind(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.wind, dtype=np.double, count=(3)), (3, ))
        arr.setflags(write=False)
        return arr
    
    @wind.setter
    def wind(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.wind, val_ptr, 3 * sizeof(c_double))
    
    @property
    def density(self):
        return self._wrapped.contents.density
    
    @density.setter
    def density(self, value):
        self._wrapped.contents.density = value
    
    @property
    def viscosity(self):
        return self._wrapped.contents.viscosity
    
    @viscosity.setter
    def viscosity(self, value):
        self._wrapped.contents.viscosity = value
    
    @property
    def expdist(self):
        return self._wrapped.contents.expdist
    
    @expdist.setter
    def expdist(self, value):
        self._wrapped.contents.expdist = value
    
    @property
    def s_mindist(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.s_mindist, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @s_mindist.setter
    def s_mindist(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.s_mindist, val_ptr, 2 * sizeof(c_double))
    
    @property
    def s_stiffness(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.s_stiffness, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @s_stiffness.setter
    def s_stiffness(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.s_stiffness, val_ptr, 2 * sizeof(c_double))
    
    @property
    def s_damping(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.s_damping, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @s_damping.setter
    def s_damping(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.s_damping, val_ptr, 2 * sizeof(c_double))
    
    @property
    def s_armature(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.s_armature, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @s_armature.setter
    def s_armature(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.s_armature, val_ptr, 2 * sizeof(c_double))
    
    @property
    def s_friction(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.s_friction, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @s_friction.setter
    def s_friction(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.s_friction, val_ptr, 2 * sizeof(c_double))
    
    @property
    def s_frictionloss(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.s_frictionloss, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @s_frictionloss.setter
    def s_frictionloss(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.s_frictionloss, val_ptr, 2 * sizeof(c_double))
    
    @property
    def s_compliance(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.s_compliance, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @s_compliance.setter
    def s_compliance(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.s_compliance, val_ptr, 2 * sizeof(c_double))
    
    @property
    def s_timeconst(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.s_timeconst, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @s_timeconst.setter
    def s_timeconst(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.s_timeconst, val_ptr, 2 * sizeof(c_double))
    
    @property
    def disableflags(self):
        return self._wrapped.contents.disableflags
    
    @disableflags.setter
    def disableflags(self, value):
        self._wrapped.contents.disableflags = value
    
    @property
    def integrator(self):
        return self._wrapped.contents.integrator
    
    @integrator.setter
    def integrator(self, value):
        self._wrapped.contents.integrator = value
    
    @property
    def collisionmode(self):
        return self._wrapped.contents.collisionmode
    
    @collisionmode.setter
    def collisionmode(self, value):
        self._wrapped.contents.collisionmode = value
    
    @property
    def algorithm(self):
        return self._wrapped.contents.algorithm
    
    @algorithm.setter
    def algorithm(self, value):
        self._wrapped.contents.algorithm = value
    
    @property
    def iterations(self):
        return self._wrapped.contents.iterations
    
    @iterations.setter
    def iterations(self, value):
        self._wrapped.contents.iterations = value
    
    @property
    def eqsoft(self):
        return self._wrapped.contents.eqsoft
    
    @eqsoft.setter
    def eqsoft(self, value):
        self._wrapped.contents.eqsoft = value
    
    @property
    def fastdiag(self):
        return self._wrapped.contents.fastdiag
    
    @fastdiag.setter
    def fastdiag(self, value):
        self._wrapped.contents.fastdiag = value
    
    @property
    def remotecontact(self):
        return self._wrapped.contents.remotecontact
    
    @remotecontact.setter
    def remotecontact(self, value):
        self._wrapped.contents.remotecontact = value
    
    @property
    def stats(self):
        return self._wrapped.contents.stats
    
    @stats.setter
    def stats(self, value):
        self._wrapped.contents.stats = value

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
    def nstack(self):
        return self._wrapped.contents.nstack
    
    @nstack.setter
    def nstack(self, value):
        self._wrapped.contents.nstack = value
    
    @property
    def nbuffer(self):
        return self._wrapped.contents.nbuffer
    
    @nbuffer.setter
    def nbuffer(self, value):
        self._wrapped.contents.nbuffer = value
    
    @property
    def pstack(self):
        return self._wrapped.contents.pstack
    
    @pstack.setter
    def pstack(self, value):
        self._wrapped.contents.pstack = value
    
    @property
    def maxstackuse(self):
        return self._wrapped.contents.maxstackuse
    
    @maxstackuse.setter
    def maxstackuse(self, value):
        self._wrapped.contents.maxstackuse = value
    
    @property
    def ne(self):
        return self._wrapped.contents.ne
    
    @ne.setter
    def ne(self, value):
        self._wrapped.contents.ne = value
    
    @property
    def nc(self):
        return self._wrapped.contents.nc
    
    @nc.setter
    def nc(self, value):
        self._wrapped.contents.nc = value
    
    @property
    def nfri(self):
        return self._wrapped.contents.nfri
    
    @nfri.setter
    def nfri(self, value):
        self._wrapped.contents.nfri = value
    
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
    def nflc(self):
        return self._wrapped.contents.nflc
    
    @nflc.setter
    def nflc(self, value):
        self._wrapped.contents.nflc = value
    
    @property
    def nstat(self):
        return self._wrapped.contents.nstat
    
    @nstat.setter
    def nstat(self, value):
        self._wrapped.contents.nstat = value
    
    @property
    def nwarning(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.nwarning, dtype=np.int, count=(7)), (7, ))
        arr.setflags(write=False)
        return arr
    
    @nwarning.setter
    def nwarning(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.nwarning, val_ptr, 7 * sizeof(c_int))
    
    @property
    def time(self):
        return self._wrapped.contents.time
    
    @time.setter
    def time(self, value):
        self._wrapped.contents.time = value
    
    @property
    def energy(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.energy, dtype=np.double, count=(2)), (2, ))
        arr.setflags(write=False)
        return arr
    
    @energy.setter
    def energy(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.energy, val_ptr, 2 * sizeof(c_double))
    
    @property
    def buffer(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.buffer, dtype=np.uint8, count=(self.nbuffer)), (self.nbuffer, ))
        arr.setflags(write=False)
        return arr
    
    @buffer.setter
    def buffer(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.buffer, val_ptr, self.nbuffer * sizeof(c_ubyte))
    
    @property
    def stack(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.stack, dtype=np.double, count=(self.nstack)), (self.nstack, ))
        arr.setflags(write=False)
        return arr
    
    @stack.setter
    def stack(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.stack, val_ptr, self.nstack * sizeof(c_double))
    
    @property
    def userdata(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.userdata, dtype=np.double, count=(self._size_src.nuserdata*1)), (self._size_src.nuserdata, 1, ))
        arr.setflags(write=False)
        return arr
    
    @userdata.setter
    def userdata(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.userdata, val_ptr, self._size_src.nuserdata*1 * sizeof(c_double))
    
    @property
    def qpos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qpos, dtype=np.double, count=(self._size_src.nq*1)), (self._size_src.nq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qpos.setter
    def qpos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qpos, val_ptr, self._size_src.nq*1 * sizeof(c_double))
    
    @property
    def qvel(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qvel, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qvel.setter
    def qvel(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qvel, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
    @property
    def qvel_next(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qvel_next, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qvel_next.setter
    def qvel_next(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qvel_next, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
    @property
    def act(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.act, dtype=np.double, count=(self._size_src.na*1)), (self._size_src.na, 1, ))
        arr.setflags(write=False)
        return arr
    
    @act.setter
    def act(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.act, val_ptr, self._size_src.na*1 * sizeof(c_double))
    
    @property
    def act_next(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.act_next, dtype=np.double, count=(self._size_src.na*1)), (self._size_src.na, 1, ))
        arr.setflags(write=False)
        return arr
    
    @act_next.setter
    def act_next(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.act_next, val_ptr, self._size_src.na*1 * sizeof(c_double))
    
    @property
    def ctrl(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.ctrl, dtype=np.double, count=(self._size_src.nu*1)), (self._size_src.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @ctrl.setter
    def ctrl(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.ctrl, val_ptr, self._size_src.nu*1 * sizeof(c_double))
    
    @property
    def qfrc_applied(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qfrc_applied, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qfrc_applied.setter
    def qfrc_applied(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qfrc_applied, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
    @property
    def xfrc_applied(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.xfrc_applied, dtype=np.double, count=(self._size_src.nbody*6)), (self._size_src.nbody, 6, ))
        arr.setflags(write=False)
        return arr
    
    @xfrc_applied.setter
    def xfrc_applied(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.xfrc_applied, val_ptr, self._size_src.nbody*6 * sizeof(c_double))
    
    @property
    def qfrc_bias(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qfrc_bias, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qfrc_bias.setter
    def qfrc_bias(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qfrc_bias, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
    @property
    def qfrc_passive(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qfrc_passive, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qfrc_passive.setter
    def qfrc_passive(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qfrc_passive, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
    @property
    def qfrc_actuation(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qfrc_actuation, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qfrc_actuation.setter
    def qfrc_actuation(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qfrc_actuation, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
    @property
    def qfrc_impulse(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qfrc_impulse, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qfrc_impulse.setter
    def qfrc_impulse(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qfrc_impulse, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
    @property
    def qfrc_constraint(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qfrc_constraint, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qfrc_constraint.setter
    def qfrc_constraint(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qfrc_constraint, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
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
    def xipos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.xipos, dtype=np.double, count=(self._size_src.nbody*3)), (self._size_src.nbody, 3, ))
        arr.setflags(write=False)
        return arr
    
    @xipos.setter
    def xipos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.xipos, val_ptr, self._size_src.nbody*3 * sizeof(c_double))
    
    @property
    def ximat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.ximat, dtype=np.double, count=(self._size_src.nbody*9)), (self._size_src.nbody, 9, ))
        arr.setflags(write=False)
        return arr
    
    @ximat.setter
    def ximat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.ximat, val_ptr, self._size_src.nbody*9 * sizeof(c_double))
    
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
    def site_xpos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.site_xpos, dtype=np.double, count=(self._size_src.nsite*3)), (self._size_src.nsite, 3, ))
        arr.setflags(write=False)
        return arr
    
    @site_xpos.setter
    def site_xpos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.site_xpos, val_ptr, self._size_src.nsite*3 * sizeof(c_double))
    
    @property
    def site_xmat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.site_xmat, dtype=np.double, count=(self._size_src.nsite*9)), (self._size_src.nsite, 9, ))
        arr.setflags(write=False)
        return arr
    
    @site_xmat.setter
    def site_xmat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.site_xmat, val_ptr, self._size_src.nsite*9 * sizeof(c_double))
    
    @property
    def com_subtree(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.com_subtree, dtype=np.double, count=(self._size_src.nbody*3)), (self._size_src.nbody, 3, ))
        arr.setflags(write=False)
        return arr
    
    @com_subtree.setter
    def com_subtree(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.com_subtree, val_ptr, self._size_src.nbody*3 * sizeof(c_double))
    
    @property
    def cdof(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cdof, dtype=np.double, count=(self._size_src.nv*6)), (self._size_src.nv, 6, ))
        arr.setflags(write=False)
        return arr
    
    @cdof.setter
    def cdof(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cdof, val_ptr, self._size_src.nv*6 * sizeof(c_double))
    
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
    def ten_wrapadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.ten_wrapadr, dtype=np.int, count=(self._size_src.ntendon*1)), (self._size_src.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @ten_wrapadr.setter
    def ten_wrapadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.ten_wrapadr, val_ptr, self._size_src.ntendon*1 * sizeof(c_int))
    
    @property
    def ten_wrapnum(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.ten_wrapnum, dtype=np.int, count=(self._size_src.ntendon*1)), (self._size_src.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @ten_wrapnum.setter
    def ten_wrapnum(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.ten_wrapnum, val_ptr, self._size_src.ntendon*1 * sizeof(c_int))
    
    @property
    def ten_length(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.ten_length, dtype=np.double, count=(self._size_src.ntendon*1)), (self._size_src.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @ten_length.setter
    def ten_length(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.ten_length, val_ptr, self._size_src.ntendon*1 * sizeof(c_double))
    
    @property
    def ten_moment(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.ten_moment, dtype=np.double, count=(self._size_src.ntendon*self._size_src.nv)), (self._size_src.ntendon, self._size_src.nv, ))
        arr.setflags(write=False)
        return arr
    
    @ten_moment.setter
    def ten_moment(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.ten_moment, val_ptr, self._size_src.ntendon*self._size_src.nv * sizeof(c_double))
    
    @property
    def wrap_obj(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.wrap_obj, dtype=np.int, count=(self._size_src.nwrap*2*1)), (self._size_src.nwrap*2, 1, ))
        arr.setflags(write=False)
        return arr
    
    @wrap_obj.setter
    def wrap_obj(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.wrap_obj, val_ptr, self._size_src.nwrap*2*1 * sizeof(c_int))
    
    @property
    def wrap_xpos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.wrap_xpos, dtype=np.double, count=(self._size_src.nwrap*2*3)), (self._size_src.nwrap*2, 3, ))
        arr.setflags(write=False)
        return arr
    
    @wrap_xpos.setter
    def wrap_xpos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.wrap_xpos, val_ptr, self._size_src.nwrap*2*3 * sizeof(c_double))
    
    @property
    def actuator_length(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_length, dtype=np.double, count=(self._size_src.nu*1)), (self._size_src.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_length.setter
    def actuator_length(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_length, val_ptr, self._size_src.nu*1 * sizeof(c_double))
    
    @property
    def actuator_moment(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_moment, dtype=np.double, count=(self._size_src.nu*self._size_src.nv)), (self._size_src.nu, self._size_src.nv, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_moment.setter
    def actuator_moment(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_moment, val_ptr, self._size_src.nu*self._size_src.nv * sizeof(c_double))
    
    @property
    def actuator_force(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_force, dtype=np.double, count=(self._size_src.nu*1)), (self._size_src.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_force.setter
    def actuator_force(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_force, val_ptr, self._size_src.nu*1 * sizeof(c_double))
    
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
    def cfrc_ext(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cfrc_ext, dtype=np.double, count=(self._size_src.nbody*6)), (self._size_src.nbody, 6, ))
        arr.setflags(write=False)
        return arr
    
    @cfrc_ext.setter
    def cfrc_ext(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cfrc_ext, val_ptr, self._size_src.nbody*6 * sizeof(c_double))
    
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
    def qD(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qD, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qD.setter
    def qD(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qD, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
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
    def qLDiagSqr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qLDiagSqr, dtype=np.double, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qLDiagSqr.setter
    def qLDiagSqr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qLDiagSqr, val_ptr, self._size_src.nv*1 * sizeof(c_double))
    
    @property
    def eq_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_id, dtype=np.int, count=(self._size_src.nemax*1)), (self._size_src.nemax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_id.setter
    def eq_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_id, val_ptr, self._size_src.nemax*1 * sizeof(c_int))
    
    @property
    def eq_err(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_err, dtype=np.double, count=(self._size_src.nemax*1)), (self._size_src.nemax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_err.setter
    def eq_err(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_err, val_ptr, self._size_src.nemax*1 * sizeof(c_double))
    
    @property
    def eq_J(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_J, dtype=np.double, count=(self._size_src.nemax*self._size_src.nv)), (self._size_src.nemax, self._size_src.nv, ))
        arr.setflags(write=False)
        return arr
    
    @eq_J.setter
    def eq_J(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_J, val_ptr, self._size_src.nemax*self._size_src.nv * sizeof(c_double))
    
    @property
    def eq_vdes(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_vdes, dtype=np.double, count=(self._size_src.nemax*1)), (self._size_src.nemax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_vdes.setter
    def eq_vdes(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_vdes, val_ptr, self._size_src.nemax*1 * sizeof(c_double))
    
    @property
    def eq_JMi(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_JMi, dtype=np.double, count=(self._size_src.nemax*self._size_src.nv)), (self._size_src.nemax, self._size_src.nv, ))
        arr.setflags(write=False)
        return arr
    
    @eq_JMi.setter
    def eq_JMi(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_JMi, val_ptr, self._size_src.nemax*self._size_src.nv * sizeof(c_double))
    
    @property
    def eq_Achol(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_Achol, dtype=np.double, count=(self._size_src.nemax*self._size_src.nemax)), (self._size_src.nemax, self._size_src.nemax, ))
        arr.setflags(write=False)
        return arr
    
    @eq_Achol.setter
    def eq_Achol(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_Achol, val_ptr, self._size_src.nemax*self._size_src.nemax * sizeof(c_double))
    
    @property
    def eq_R(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_R, dtype=np.double, count=(self._size_src.nemax*1)), (self._size_src.nemax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_R.setter
    def eq_R(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_R, val_ptr, self._size_src.nemax*1 * sizeof(c_double))
    
    @property
    def fri_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.fri_id, dtype=np.int, count=(self._size_src.nv*1)), (self._size_src.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @fri_id.setter
    def fri_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.fri_id, val_ptr, self._size_src.nv*1 * sizeof(c_int))
    
    @property
    def lim_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.lim_id, dtype=np.int, count=(self._size_src.nlmax*1)), (self._size_src.nlmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @lim_id.setter
    def lim_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.lim_id, val_ptr, self._size_src.nlmax*1 * sizeof(c_int))
    
    @property
    def con_id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.con_id, dtype=np.int, count=(self._size_src.ncmax*1)), (self._size_src.ncmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @con_id.setter
    def con_id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.con_id, val_ptr, self._size_src.ncmax*1 * sizeof(c_int))
    
    @property
    def flc_signature(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.flc_signature, dtype=np.int, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @flc_signature.setter
    def flc_signature(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.flc_signature, val_ptr, self._size_src.njmax*1 * sizeof(c_int))
    
    @property
    def flc_J(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.flc_J, dtype=np.double, count=(self._size_src.njmax*self._size_src.nv)), (self._size_src.njmax, self._size_src.nv, ))
        arr.setflags(write=False)
        return arr
    
    @flc_J.setter
    def flc_J(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.flc_J, val_ptr, self._size_src.njmax*self._size_src.nv * sizeof(c_double))
    
    @property
    def flc_ek(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.flc_ek, dtype=np.double, count=(self._size_src.njmax*2)), (self._size_src.njmax, 2, ))
        arr.setflags(write=False)
        return arr
    
    @flc_ek.setter
    def flc_ek(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.flc_ek, val_ptr, self._size_src.njmax*2 * sizeof(c_double))
    
    @property
    def flc_A(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.flc_A, dtype=np.double, count=(self._size_src.njmax*self._size_src.njmax)), (self._size_src.njmax, self._size_src.njmax, ))
        arr.setflags(write=False)
        return arr
    
    @flc_A.setter
    def flc_A(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.flc_A, val_ptr, self._size_src.njmax*self._size_src.njmax * sizeof(c_double))
    
    @property
    def flc_R(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.flc_R, dtype=np.double, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @flc_R.setter
    def flc_R(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.flc_R, val_ptr, self._size_src.njmax*1 * sizeof(c_double))
    
    @property
    def flc_vdes(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.flc_vdes, dtype=np.double, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @flc_vdes.setter
    def flc_vdes(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.flc_vdes, val_ptr, self._size_src.njmax*1 * sizeof(c_double))
    
    @property
    def flc_b(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.flc_b, dtype=np.double, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @flc_b.setter
    def flc_b(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.flc_b, val_ptr, self._size_src.njmax*1 * sizeof(c_double))
    
    @property
    def flc_f(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.flc_f, dtype=np.double, count=(self._size_src.njmax*1)), (self._size_src.njmax, 1, ))
        arr.setflags(write=False)
        return arr
    
    @flc_f.setter
    def flc_f(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.flc_f, val_ptr, self._size_src.njmax*1 * sizeof(c_double))

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
    def nq(self):
        return self._wrapped.contents.nq
    
    @nq.setter
    def nq(self, value):
        self._wrapped.contents.nq = value
    
    @property
    def nv(self):
        return self._wrapped.contents.nv
    
    @nv.setter
    def nv(self, value):
        self._wrapped.contents.nv = value
    
    @property
    def nu(self):
        return self._wrapped.contents.nu
    
    @nu.setter
    def nu(self, value):
        self._wrapped.contents.nu = value
    
    @property
    def na(self):
        return self._wrapped.contents.na
    
    @na.setter
    def na(self, value):
        self._wrapped.contents.na = value
    
    @property
    def nbody(self):
        return self._wrapped.contents.nbody
    
    @nbody.setter
    def nbody(self, value):
        self._wrapped.contents.nbody = value
    
    @property
    def njnt(self):
        return self._wrapped.contents.njnt
    
    @njnt.setter
    def njnt(self, value):
        self._wrapped.contents.njnt = value
    
    @property
    def ngeom(self):
        return self._wrapped.contents.ngeom
    
    @ngeom.setter
    def ngeom(self, value):
        self._wrapped.contents.ngeom = value
    
    @property
    def nsite(self):
        return self._wrapped.contents.nsite
    
    @nsite.setter
    def nsite(self, value):
        self._wrapped.contents.nsite = value
    
    @property
    def ncam(self):
        return self._wrapped.contents.ncam
    
    @ncam.setter
    def ncam(self, value):
        self._wrapped.contents.ncam = value
    
    @property
    def nmesh(self):
        return self._wrapped.contents.nmesh
    
    @nmesh.setter
    def nmesh(self, value):
        self._wrapped.contents.nmesh = value
    
    @property
    def nmeshvert(self):
        return self._wrapped.contents.nmeshvert
    
    @nmeshvert.setter
    def nmeshvert(self, value):
        self._wrapped.contents.nmeshvert = value
    
    @property
    def nmeshface(self):
        return self._wrapped.contents.nmeshface
    
    @nmeshface.setter
    def nmeshface(self, value):
        self._wrapped.contents.nmeshface = value
    
    @property
    def nmeshgraph(self):
        return self._wrapped.contents.nmeshgraph
    
    @nmeshgraph.setter
    def nmeshgraph(self, value):
        self._wrapped.contents.nmeshgraph = value
    
    @property
    def nhfield(self):
        return self._wrapped.contents.nhfield
    
    @nhfield.setter
    def nhfield(self, value):
        self._wrapped.contents.nhfield = value
    
    @property
    def nhfielddata(self):
        return self._wrapped.contents.nhfielddata
    
    @nhfielddata.setter
    def nhfielddata(self, value):
        self._wrapped.contents.nhfielddata = value
    
    @property
    def ncpair(self):
        return self._wrapped.contents.ncpair
    
    @ncpair.setter
    def ncpair(self, value):
        self._wrapped.contents.ncpair = value
    
    @property
    def neq(self):
        return self._wrapped.contents.neq
    
    @neq.setter
    def neq(self, value):
        self._wrapped.contents.neq = value
    
    @property
    def ntendon(self):
        return self._wrapped.contents.ntendon
    
    @ntendon.setter
    def ntendon(self, value):
        self._wrapped.contents.ntendon = value
    
    @property
    def nwrap(self):
        return self._wrapped.contents.nwrap
    
    @nwrap.setter
    def nwrap(self, value):
        self._wrapped.contents.nwrap = value
    
    @property
    def nnumeric(self):
        return self._wrapped.contents.nnumeric
    
    @nnumeric.setter
    def nnumeric(self, value):
        self._wrapped.contents.nnumeric = value
    
    @property
    def nnumericdata(self):
        return self._wrapped.contents.nnumericdata
    
    @nnumericdata.setter
    def nnumericdata(self, value):
        self._wrapped.contents.nnumericdata = value
    
    @property
    def ntext(self):
        return self._wrapped.contents.ntext
    
    @ntext.setter
    def ntext(self, value):
        self._wrapped.contents.ntext = value
    
    @property
    def ntextdata(self):
        return self._wrapped.contents.ntextdata
    
    @ntextdata.setter
    def ntextdata(self, value):
        self._wrapped.contents.ntextdata = value
    
    @property
    def nuser_body(self):
        return self._wrapped.contents.nuser_body
    
    @nuser_body.setter
    def nuser_body(self, value):
        self._wrapped.contents.nuser_body = value
    
    @property
    def nuser_jnt(self):
        return self._wrapped.contents.nuser_jnt
    
    @nuser_jnt.setter
    def nuser_jnt(self, value):
        self._wrapped.contents.nuser_jnt = value
    
    @property
    def nuser_geom(self):
        return self._wrapped.contents.nuser_geom
    
    @nuser_geom.setter
    def nuser_geom(self, value):
        self._wrapped.contents.nuser_geom = value
    
    @property
    def nuser_site(self):
        return self._wrapped.contents.nuser_site
    
    @nuser_site.setter
    def nuser_site(self, value):
        self._wrapped.contents.nuser_site = value
    
    @property
    def nuser_eq(self):
        return self._wrapped.contents.nuser_eq
    
    @nuser_eq.setter
    def nuser_eq(self, value):
        self._wrapped.contents.nuser_eq = value
    
    @property
    def nuser_tendon(self):
        return self._wrapped.contents.nuser_tendon
    
    @nuser_tendon.setter
    def nuser_tendon(self, value):
        self._wrapped.contents.nuser_tendon = value
    
    @property
    def nuser_actuator(self):
        return self._wrapped.contents.nuser_actuator
    
    @nuser_actuator.setter
    def nuser_actuator(self, value):
        self._wrapped.contents.nuser_actuator = value
    
    @property
    def nnames(self):
        return self._wrapped.contents.nnames
    
    @nnames.setter
    def nnames(self, value):
        self._wrapped.contents.nnames = value
    
    @property
    def nM(self):
        return self._wrapped.contents.nM
    
    @nM.setter
    def nM(self, value):
        self._wrapped.contents.nM = value
    
    @property
    def nemax(self):
        return self._wrapped.contents.nemax
    
    @nemax.setter
    def nemax(self, value):
        self._wrapped.contents.nemax = value
    
    @property
    def nlmax(self):
        return self._wrapped.contents.nlmax
    
    @nlmax.setter
    def nlmax(self, value):
        self._wrapped.contents.nlmax = value
    
    @property
    def ncmax(self):
        return self._wrapped.contents.ncmax
    
    @ncmax.setter
    def ncmax(self, value):
        self._wrapped.contents.ncmax = value
    
    @property
    def njmax(self):
        return self._wrapped.contents.njmax
    
    @njmax.setter
    def njmax(self, value):
        self._wrapped.contents.njmax = value
    
    @property
    def nctotmax(self):
        return self._wrapped.contents.nctotmax
    
    @nctotmax.setter
    def nctotmax(self, value):
        self._wrapped.contents.nctotmax = value
    
    @property
    def nstack(self):
        return self._wrapped.contents.nstack
    
    @nstack.setter
    def nstack(self, value):
        self._wrapped.contents.nstack = value
    
    @property
    def nuserdata(self):
        return self._wrapped.contents.nuserdata
    
    @nuserdata.setter
    def nuserdata(self, value):
        self._wrapped.contents.nuserdata = value
    
    @property
    def nbuffer(self):
        return self._wrapped.contents.nbuffer
    
    @nbuffer.setter
    def nbuffer(self, value):
        self._wrapped.contents.nbuffer = value
    
    @property
    def buffer(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.buffer, dtype=np.uint8, count=(self.nbuffer)), (self.nbuffer, ))
        arr.setflags(write=False)
        return arr
    
    @buffer.setter
    def buffer(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.buffer, val_ptr, self.nbuffer * sizeof(c_ubyte))
    
    @property
    def qpos0(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qpos0, dtype=np.double, count=(self.nq*1)), (self.nq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qpos0.setter
    def qpos0(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qpos0, val_ptr, self.nq*1 * sizeof(c_double))
    
    @property
    def qpos_spring(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.qpos_spring, dtype=np.double, count=(self.nq*1)), (self.nq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @qpos_spring.setter
    def qpos_spring(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.qpos_spring, val_ptr, self.nq*1 * sizeof(c_double))
    
    @property
    def body_parentid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_parentid, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_parentid.setter
    def body_parentid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_parentid, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_rootid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_rootid, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_rootid.setter
    def body_rootid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_rootid, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_weldid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_weldid, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_weldid.setter
    def body_weldid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_weldid, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_jntnum(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_jntnum, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_jntnum.setter
    def body_jntnum(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_jntnum, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_jntadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_jntadr, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_jntadr.setter
    def body_jntadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_jntadr, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_dofnum(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_dofnum, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_dofnum.setter
    def body_dofnum(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_dofnum, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_dofadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_dofadr, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_dofadr.setter
    def body_dofadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_dofadr, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_geomnum(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_geomnum, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_geomnum.setter
    def body_geomnum(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_geomnum, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_geomadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_geomadr, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_geomadr.setter
    def body_geomadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.body_geomadr, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def body_gravcomp(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_gravcomp, dtype=np.uint8, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @body_gravcomp.setter
    def body_gravcomp(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.body_gravcomp, val_ptr, self.nbody*1 * sizeof(c_ubyte))
    
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
    def body_ipos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_ipos, dtype=np.double, count=(self.nbody*3)), (self.nbody, 3, ))
        arr.setflags(write=False)
        return arr
    
    @body_ipos.setter
    def body_ipos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_ipos, val_ptr, self.nbody*3 * sizeof(c_double))
    
    @property
    def body_iquat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_iquat, dtype=np.double, count=(self.nbody*4)), (self.nbody, 4, ))
        arr.setflags(write=False)
        return arr
    
    @body_iquat.setter
    def body_iquat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_iquat, val_ptr, self.nbody*4 * sizeof(c_double))
    
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
    def body_invweight0(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_invweight0, dtype=np.double, count=(self.nbody*2)), (self.nbody, 2, ))
        arr.setflags(write=False)
        return arr
    
    @body_invweight0.setter
    def body_invweight0(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_invweight0, val_ptr, self.nbody*2 * sizeof(c_double))
    
    @property
    def body_user(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.body_user, dtype=np.double, count=(self.nbody*self.nuser_body)), (self.nbody, self.nuser_body, ))
        arr.setflags(write=False)
        return arr
    
    @body_user.setter
    def body_user(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.body_user, val_ptr, self.nbody*self.nuser_body * sizeof(c_double))
    
    @property
    def jnt_type(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_type, dtype=np.int, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_type.setter
    def jnt_type(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.jnt_type, val_ptr, self.njnt*1 * sizeof(c_int))
    
    @property
    def jnt_qposadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_qposadr, dtype=np.int, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_qposadr.setter
    def jnt_qposadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.jnt_qposadr, val_ptr, self.njnt*1 * sizeof(c_int))
    
    @property
    def jnt_dofadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_dofadr, dtype=np.int, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_dofadr.setter
    def jnt_dofadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.jnt_dofadr, val_ptr, self.njnt*1 * sizeof(c_int))
    
    @property
    def jnt_bodyid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_bodyid, dtype=np.int, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_bodyid.setter
    def jnt_bodyid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.jnt_bodyid, val_ptr, self.njnt*1 * sizeof(c_int))
    
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
    def jnt_stiffness(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_stiffness, dtype=np.double, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_stiffness.setter
    def jnt_stiffness(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_stiffness, val_ptr, self.njnt*1 * sizeof(c_double))
    
    @property
    def jnt_range(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_range, dtype=np.double, count=(self.njnt*2)), (self.njnt, 2, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_range.setter
    def jnt_range(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_range, val_ptr, self.njnt*2 * sizeof(c_double))
    
    @property
    def jnt_compliance(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_compliance, dtype=np.double, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_compliance.setter
    def jnt_compliance(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_compliance, val_ptr, self.njnt*1 * sizeof(c_double))
    
    @property
    def jnt_timeconst(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_timeconst, dtype=np.double, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_timeconst.setter
    def jnt_timeconst(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_timeconst, val_ptr, self.njnt*1 * sizeof(c_double))
    
    @property
    def jnt_mindist(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_mindist, dtype=np.double, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_mindist.setter
    def jnt_mindist(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_mindist, val_ptr, self.njnt*1 * sizeof(c_double))
    
    @property
    def jnt_user(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.jnt_user, dtype=np.double, count=(self.njnt*self.nuser_jnt)), (self.njnt, self.nuser_jnt, ))
        arr.setflags(write=False)
        return arr
    
    @jnt_user.setter
    def jnt_user(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.jnt_user, val_ptr, self.njnt*self.nuser_jnt * sizeof(c_double))
    
    @property
    def dof_bodyid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_bodyid, dtype=np.int, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_bodyid.setter
    def dof_bodyid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.dof_bodyid, val_ptr, self.nv*1 * sizeof(c_int))
    
    @property
    def dof_jntid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_jntid, dtype=np.int, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_jntid.setter
    def dof_jntid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.dof_jntid, val_ptr, self.nv*1 * sizeof(c_int))
    
    @property
    def dof_parentid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_parentid, dtype=np.int, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_parentid.setter
    def dof_parentid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.dof_parentid, val_ptr, self.nv*1 * sizeof(c_int))
    
    @property
    def dof_Madr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_Madr, dtype=np.int, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_Madr.setter
    def dof_Madr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.dof_Madr, val_ptr, self.nv*1 * sizeof(c_int))
    
    @property
    def dof_isfrictional(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_isfrictional, dtype=np.uint8, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_isfrictional.setter
    def dof_isfrictional(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.dof_isfrictional, val_ptr, self.nv*1 * sizeof(c_ubyte))
    
    @property
    def dof_armature(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_armature, dtype=np.double, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_armature.setter
    def dof_armature(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.dof_armature, val_ptr, self.nv*1 * sizeof(c_double))
    
    @property
    def dof_damping(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_damping, dtype=np.double, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_damping.setter
    def dof_damping(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.dof_damping, val_ptr, self.nv*1 * sizeof(c_double))
    
    @property
    def dof_frictionloss(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_frictionloss, dtype=np.double, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_frictionloss.setter
    def dof_frictionloss(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.dof_frictionloss, val_ptr, self.nv*1 * sizeof(c_double))
    
    @property
    def dof_maxvel(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_maxvel, dtype=np.double, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_maxvel.setter
    def dof_maxvel(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.dof_maxvel, val_ptr, self.nv*1 * sizeof(c_double))
    
    @property
    def dof_invweight0(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.dof_invweight0, dtype=np.double, count=(self.nv*1)), (self.nv, 1, ))
        arr.setflags(write=False)
        return arr
    
    @dof_invweight0.setter
    def dof_invweight0(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.dof_invweight0, val_ptr, self.nv*1 * sizeof(c_double))
    
    @property
    def geom_type(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_type, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_type.setter
    def geom_type(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.geom_type, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def geom_contype(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_contype, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_contype.setter
    def geom_contype(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.geom_contype, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def geom_conaffinity(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_conaffinity, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_conaffinity.setter
    def geom_conaffinity(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.geom_conaffinity, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def geom_condim(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_condim, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_condim.setter
    def geom_condim(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.geom_condim, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def geom_bodyid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_bodyid, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_bodyid.setter
    def geom_bodyid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.geom_bodyid, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def geom_dataid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_dataid, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_dataid.setter
    def geom_dataid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.geom_dataid, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def geom_group(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_group, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_group.setter
    def geom_group(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.geom_group, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def geom_size(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_size, dtype=np.double, count=(self.ngeom*3)), (self.ngeom, 3, ))
        arr.setflags(write=False)
        return arr
    
    @geom_size.setter
    def geom_size(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_size, val_ptr, self.ngeom*3 * sizeof(c_double))
    
    @property
    def geom_rbound(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_rbound, dtype=np.double, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_rbound.setter
    def geom_rbound(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_rbound, val_ptr, self.ngeom*1 * sizeof(c_double))
    
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
    def geom_friction(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_friction, dtype=np.double, count=(self.ngeom*3)), (self.ngeom, 3, ))
        arr.setflags(write=False)
        return arr
    
    @geom_friction.setter
    def geom_friction(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_friction, val_ptr, self.ngeom*3 * sizeof(c_double))
    
    @property
    def geom_compliance(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_compliance, dtype=np.double, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_compliance.setter
    def geom_compliance(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_compliance, val_ptr, self.ngeom*1 * sizeof(c_double))
    
    @property
    def geom_timeconst(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_timeconst, dtype=np.double, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_timeconst.setter
    def geom_timeconst(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_timeconst, val_ptr, self.ngeom*1 * sizeof(c_double))
    
    @property
    def geom_mindist(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_mindist, dtype=np.double, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @geom_mindist.setter
    def geom_mindist(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_mindist, val_ptr, self.ngeom*1 * sizeof(c_double))
    
    @property
    def geom_user(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_user, dtype=np.double, count=(self.ngeom*self.nuser_geom)), (self.ngeom, self.nuser_geom, ))
        arr.setflags(write=False)
        return arr
    
    @geom_user.setter
    def geom_user(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.geom_user, val_ptr, self.ngeom*self.nuser_geom * sizeof(c_double))
    
    @property
    def geom_rgba(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.geom_rgba, dtype=np.float, count=(self.ngeom*4)), (self.ngeom, 4, ))
        arr.setflags(write=False)
        return arr
    
    @geom_rgba.setter
    def geom_rgba(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.geom_rgba, val_ptr, self.ngeom*4 * sizeof(c_float))
    
    @property
    def site_bodyid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.site_bodyid, dtype=np.int, count=(self.nsite*1)), (self.nsite, 1, ))
        arr.setflags(write=False)
        return arr
    
    @site_bodyid.setter
    def site_bodyid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.site_bodyid, val_ptr, self.nsite*1 * sizeof(c_int))
    
    @property
    def site_group(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.site_group, dtype=np.int, count=(self.nsite*1)), (self.nsite, 1, ))
        arr.setflags(write=False)
        return arr
    
    @site_group.setter
    def site_group(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.site_group, val_ptr, self.nsite*1 * sizeof(c_int))
    
    @property
    def site_pos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.site_pos, dtype=np.double, count=(self.nsite*3)), (self.nsite, 3, ))
        arr.setflags(write=False)
        return arr
    
    @site_pos.setter
    def site_pos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.site_pos, val_ptr, self.nsite*3 * sizeof(c_double))
    
    @property
    def site_quat(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.site_quat, dtype=np.double, count=(self.nsite*4)), (self.nsite, 4, ))
        arr.setflags(write=False)
        return arr
    
    @site_quat.setter
    def site_quat(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.site_quat, val_ptr, self.nsite*4 * sizeof(c_double))
    
    @property
    def site_user(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.site_user, dtype=np.double, count=(self.nsite*self.nuser_site)), (self.nsite, self.nuser_site, ))
        arr.setflags(write=False)
        return arr
    
    @site_user.setter
    def site_user(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.site_user, val_ptr, self.nsite*self.nuser_site * sizeof(c_double))
    
    @property
    def mesh_faceadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mesh_faceadr, dtype=np.int, count=(self.nmesh*1)), (self.nmesh, 1, ))
        arr.setflags(write=False)
        return arr
    
    @mesh_faceadr.setter
    def mesh_faceadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.mesh_faceadr, val_ptr, self.nmesh*1 * sizeof(c_int))
    
    @property
    def mesh_facenum(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mesh_facenum, dtype=np.int, count=(self.nmesh*1)), (self.nmesh, 1, ))
        arr.setflags(write=False)
        return arr
    
    @mesh_facenum.setter
    def mesh_facenum(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.mesh_facenum, val_ptr, self.nmesh*1 * sizeof(c_int))
    
    @property
    def mesh_vertadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mesh_vertadr, dtype=np.int, count=(self.nmesh*1)), (self.nmesh, 1, ))
        arr.setflags(write=False)
        return arr
    
    @mesh_vertadr.setter
    def mesh_vertadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.mesh_vertadr, val_ptr, self.nmesh*1 * sizeof(c_int))
    
    @property
    def mesh_vertnum(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mesh_vertnum, dtype=np.int, count=(self.nmesh*1)), (self.nmesh, 1, ))
        arr.setflags(write=False)
        return arr
    
    @mesh_vertnum.setter
    def mesh_vertnum(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.mesh_vertnum, val_ptr, self.nmesh*1 * sizeof(c_int))
    
    @property
    def mesh_graphadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mesh_graphadr, dtype=np.int, count=(self.nmesh*1)), (self.nmesh, 1, ))
        arr.setflags(write=False)
        return arr
    
    @mesh_graphadr.setter
    def mesh_graphadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.mesh_graphadr, val_ptr, self.nmesh*1 * sizeof(c_int))
    
    @property
    def mesh_vert(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mesh_vert, dtype=np.float, count=(self.nmeshvert*3)), (self.nmeshvert, 3, ))
        arr.setflags(write=False)
        return arr
    
    @mesh_vert.setter
    def mesh_vert(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.mesh_vert, val_ptr, self.nmeshvert*3 * sizeof(c_float))
    
    @property
    def mesh_face(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mesh_face, dtype=np.int, count=(self.nmeshface*3)), (self.nmeshface, 3, ))
        arr.setflags(write=False)
        return arr
    
    @mesh_face.setter
    def mesh_face(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.mesh_face, val_ptr, self.nmeshface*3 * sizeof(c_int))
    
    @property
    def mesh_graph(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.mesh_graph, dtype=np.int, count=(self.nmeshgraph*1)), (self.nmeshgraph, 1, ))
        arr.setflags(write=False)
        return arr
    
    @mesh_graph.setter
    def mesh_graph(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.mesh_graph, val_ptr, self.nmeshgraph*1 * sizeof(c_int))
    
    @property
    def hfield_nrow(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.hfield_nrow, dtype=np.int, count=(self.nhfield*1)), (self.nhfield, 1, ))
        arr.setflags(write=False)
        return arr
    
    @hfield_nrow.setter
    def hfield_nrow(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.hfield_nrow, val_ptr, self.nhfield*1 * sizeof(c_int))
    
    @property
    def hfield_ncol(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.hfield_ncol, dtype=np.int, count=(self.nhfield*1)), (self.nhfield, 1, ))
        arr.setflags(write=False)
        return arr
    
    @hfield_ncol.setter
    def hfield_ncol(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.hfield_ncol, val_ptr, self.nhfield*1 * sizeof(c_int))
    
    @property
    def hfield_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.hfield_adr, dtype=np.int, count=(self.nhfield*1)), (self.nhfield, 1, ))
        arr.setflags(write=False)
        return arr
    
    @hfield_adr.setter
    def hfield_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.hfield_adr, val_ptr, self.nhfield*1 * sizeof(c_int))
    
    @property
    def hfield_geomid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.hfield_geomid, dtype=np.int, count=(self.nhfield*1)), (self.nhfield, 1, ))
        arr.setflags(write=False)
        return arr
    
    @hfield_geomid.setter
    def hfield_geomid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.hfield_geomid, val_ptr, self.nhfield*1 * sizeof(c_int))
    
    @property
    def hfield_dir(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.hfield_dir, dtype=np.uint8, count=(self.nhfield*1)), (self.nhfield, 1, ))
        arr.setflags(write=False)
        return arr
    
    @hfield_dir.setter
    def hfield_dir(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.hfield_dir, val_ptr, self.nhfield*1 * sizeof(c_ubyte))
    
    @property
    def hfield_data(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.hfield_data, dtype=np.float, count=(self.nhfielddata*1)), (self.nhfielddata, 1, ))
        arr.setflags(write=False)
        return arr
    
    @hfield_data.setter
    def hfield_data(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_float))
        memmove(self._wrapped.contents.hfield_data, val_ptr, self.nhfielddata*1 * sizeof(c_float))
    
    @property
    def pair_dim(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_dim, dtype=np.int, count=(self.ncpair*1)), (self.ncpair, 1, ))
        arr.setflags(write=False)
        return arr
    
    @pair_dim.setter
    def pair_dim(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.pair_dim, val_ptr, self.ncpair*1 * sizeof(c_int))
    
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
    def pair_compliance(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_compliance, dtype=np.double, count=(self.ncpair*1)), (self.ncpair, 1, ))
        arr.setflags(write=False)
        return arr
    
    @pair_compliance.setter
    def pair_compliance(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.pair_compliance, val_ptr, self.ncpair*1 * sizeof(c_double))
    
    @property
    def pair_timeconst(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_timeconst, dtype=np.double, count=(self.ncpair*1)), (self.ncpair, 1, ))
        arr.setflags(write=False)
        return arr
    
    @pair_timeconst.setter
    def pair_timeconst(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.pair_timeconst, val_ptr, self.ncpair*1 * sizeof(c_double))
    
    @property
    def pair_mindist(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_mindist, dtype=np.double, count=(self.ncpair*1)), (self.ncpair, 1, ))
        arr.setflags(write=False)
        return arr
    
    @pair_mindist.setter
    def pair_mindist(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.pair_mindist, val_ptr, self.ncpair*1 * sizeof(c_double))
    
    @property
    def pair_friction(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.pair_friction, dtype=np.double, count=(self.ncpair*5)), (self.ncpair, 5, ))
        arr.setflags(write=False)
        return arr
    
    @pair_friction.setter
    def pair_friction(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.pair_friction, val_ptr, self.ncpair*5 * sizeof(c_double))
    
    @property
    def eq_type(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_type, dtype=np.int, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_type.setter
    def eq_type(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_type, val_ptr, self.neq*1 * sizeof(c_int))
    
    @property
    def eq_obj1type(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_obj1type, dtype=np.int, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_obj1type.setter
    def eq_obj1type(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_obj1type, val_ptr, self.neq*1 * sizeof(c_int))
    
    @property
    def eq_obj2type(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_obj2type, dtype=np.int, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_obj2type.setter
    def eq_obj2type(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_obj2type, val_ptr, self.neq*1 * sizeof(c_int))
    
    @property
    def eq_obj1id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_obj1id, dtype=np.int, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_obj1id.setter
    def eq_obj1id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_obj1id, val_ptr, self.neq*1 * sizeof(c_int))
    
    @property
    def eq_obj2id(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_obj2id, dtype=np.int, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_obj2id.setter
    def eq_obj2id(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_obj2id, val_ptr, self.neq*1 * sizeof(c_int))
    
    @property
    def eq_size(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_size, dtype=np.int, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_size.setter
    def eq_size(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_size, val_ptr, self.neq*1 * sizeof(c_int))
    
    @property
    def eq_ndata(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_ndata, dtype=np.int, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_ndata.setter
    def eq_ndata(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.eq_ndata, val_ptr, self.neq*1 * sizeof(c_int))
    
    @property
    def eq_isactive(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_isactive, dtype=np.uint8, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_isactive.setter
    def eq_isactive(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.eq_isactive, val_ptr, self.neq*1 * sizeof(c_ubyte))
    
    @property
    def eq_compliance(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_compliance, dtype=np.double, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_compliance.setter
    def eq_compliance(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_compliance, val_ptr, self.neq*1 * sizeof(c_double))
    
    @property
    def eq_timeconst(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_timeconst, dtype=np.double, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @eq_timeconst.setter
    def eq_timeconst(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_timeconst, val_ptr, self.neq*1 * sizeof(c_double))
    
    @property
    def eq_data(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_data, dtype=np.double, count=(self.neq*6)), (self.neq, 6, ))
        arr.setflags(write=False)
        return arr
    
    @eq_data.setter
    def eq_data(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_data, val_ptr, self.neq*6 * sizeof(c_double))
    
    @property
    def eq_user(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.eq_user, dtype=np.double, count=(self.neq*self.nuser_eq)), (self.neq, self.nuser_eq, ))
        arr.setflags(write=False)
        return arr
    
    @eq_user.setter
    def eq_user(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.eq_user, val_ptr, self.neq*self.nuser_eq * sizeof(c_double))
    
    @property
    def tendon_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_adr, dtype=np.int, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_adr.setter
    def tendon_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.tendon_adr, val_ptr, self.ntendon*1 * sizeof(c_int))
    
    @property
    def tendon_num(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_num, dtype=np.int, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_num.setter
    def tendon_num(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.tendon_num, val_ptr, self.ntendon*1 * sizeof(c_int))
    
    @property
    def tendon_islimited(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_islimited, dtype=np.uint8, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_islimited.setter
    def tendon_islimited(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.tendon_islimited, val_ptr, self.ntendon*1 * sizeof(c_ubyte))
    
    @property
    def tendon_compliance(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_compliance, dtype=np.double, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_compliance.setter
    def tendon_compliance(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_compliance, val_ptr, self.ntendon*1 * sizeof(c_double))
    
    @property
    def tendon_timeconst(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_timeconst, dtype=np.double, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_timeconst.setter
    def tendon_timeconst(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_timeconst, val_ptr, self.ntendon*1 * sizeof(c_double))
    
    @property
    def tendon_range(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_range, dtype=np.double, count=(self.ntendon*2)), (self.ntendon, 2, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_range.setter
    def tendon_range(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_range, val_ptr, self.ntendon*2 * sizeof(c_double))
    
    @property
    def tendon_mindist(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_mindist, dtype=np.double, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_mindist.setter
    def tendon_mindist(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_mindist, val_ptr, self.ntendon*1 * sizeof(c_double))
    
    @property
    def tendon_stiffness(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_stiffness, dtype=np.double, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_stiffness.setter
    def tendon_stiffness(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_stiffness, val_ptr, self.ntendon*1 * sizeof(c_double))
    
    @property
    def tendon_damping(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_damping, dtype=np.double, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_damping.setter
    def tendon_damping(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_damping, val_ptr, self.ntendon*1 * sizeof(c_double))
    
    @property
    def tendon_lengthspring(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_lengthspring, dtype=np.double, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_lengthspring.setter
    def tendon_lengthspring(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_lengthspring, val_ptr, self.ntendon*1 * sizeof(c_double))
    
    @property
    def tendon_invweight0(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_invweight0, dtype=np.double, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_invweight0.setter
    def tendon_invweight0(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_invweight0, val_ptr, self.ntendon*1 * sizeof(c_double))
    
    @property
    def tendon_user(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.tendon_user, dtype=np.double, count=(self.ntendon*self.nuser_tendon)), (self.ntendon, self.nuser_tendon, ))
        arr.setflags(write=False)
        return arr
    
    @tendon_user.setter
    def tendon_user(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.tendon_user, val_ptr, self.ntendon*self.nuser_tendon * sizeof(c_double))
    
    @property
    def wrap_type(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.wrap_type, dtype=np.int, count=(self.nwrap*1)), (self.nwrap, 1, ))
        arr.setflags(write=False)
        return arr
    
    @wrap_type.setter
    def wrap_type(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.wrap_type, val_ptr, self.nwrap*1 * sizeof(c_int))
    
    @property
    def wrap_objid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.wrap_objid, dtype=np.int, count=(self.nwrap*1)), (self.nwrap, 1, ))
        arr.setflags(write=False)
        return arr
    
    @wrap_objid.setter
    def wrap_objid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.wrap_objid, val_ptr, self.nwrap*1 * sizeof(c_int))
    
    @property
    def wrap_prm(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.wrap_prm, dtype=np.double, count=(self.nwrap*1)), (self.nwrap, 1, ))
        arr.setflags(write=False)
        return arr
    
    @wrap_prm.setter
    def wrap_prm(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.wrap_prm, val_ptr, self.nwrap*1 * sizeof(c_double))
    
    @property
    def actuator_dyntype(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_dyntype, dtype=np.int, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_dyntype.setter
    def actuator_dyntype(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.actuator_dyntype, val_ptr, self.nu*1 * sizeof(c_int))
    
    @property
    def actuator_trntype(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_trntype, dtype=np.int, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_trntype.setter
    def actuator_trntype(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.actuator_trntype, val_ptr, self.nu*1 * sizeof(c_int))
    
    @property
    def actuator_gaintype(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_gaintype, dtype=np.int, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_gaintype.setter
    def actuator_gaintype(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.actuator_gaintype, val_ptr, self.nu*1 * sizeof(c_int))
    
    @property
    def actuator_biastype(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_biastype, dtype=np.int, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_biastype.setter
    def actuator_biastype(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.actuator_biastype, val_ptr, self.nu*1 * sizeof(c_int))
    
    @property
    def actuator_trnid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_trnid, dtype=np.int, count=(self.nu*2)), (self.nu, 2, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_trnid.setter
    def actuator_trnid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.actuator_trnid, val_ptr, self.nu*2 * sizeof(c_int))
    
    @property
    def actuator_isctrllimited(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_isctrllimited, dtype=np.uint8, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_isctrllimited.setter
    def actuator_isctrllimited(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.actuator_isctrllimited, val_ptr, self.nu*1 * sizeof(c_ubyte))
    
    @property
    def actuator_isforcelimited(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_isforcelimited, dtype=np.uint8, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_isforcelimited.setter
    def actuator_isforcelimited(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_ubyte))
        memmove(self._wrapped.contents.actuator_isforcelimited, val_ptr, self.nu*1 * sizeof(c_ubyte))
    
    @property
    def actuator_dynprm(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_dynprm, dtype=np.double, count=(self.nu*10)), (self.nu, 10, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_dynprm.setter
    def actuator_dynprm(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_dynprm, val_ptr, self.nu*10 * sizeof(c_double))
    
    @property
    def actuator_trnprm(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_trnprm, dtype=np.double, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_trnprm.setter
    def actuator_trnprm(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_trnprm, val_ptr, self.nu*1 * sizeof(c_double))
    
    @property
    def actuator_gainprm(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_gainprm, dtype=np.double, count=(self.nu*5)), (self.nu, 5, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_gainprm.setter
    def actuator_gainprm(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_gainprm, val_ptr, self.nu*5 * sizeof(c_double))
    
    @property
    def actuator_biasprm(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_biasprm, dtype=np.double, count=(self.nu*3)), (self.nu, 3, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_biasprm.setter
    def actuator_biasprm(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_biasprm, val_ptr, self.nu*3 * sizeof(c_double))
    
    @property
    def actuator_ctrlrange(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_ctrlrange, dtype=np.double, count=(self.nu*2)), (self.nu, 2, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_ctrlrange.setter
    def actuator_ctrlrange(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_ctrlrange, val_ptr, self.nu*2 * sizeof(c_double))
    
    @property
    def actuator_forcerange(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_forcerange, dtype=np.double, count=(self.nu*2)), (self.nu, 2, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_forcerange.setter
    def actuator_forcerange(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_forcerange, val_ptr, self.nu*2 * sizeof(c_double))
    
    @property
    def actuator_invweight0(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_invweight0, dtype=np.double, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_invweight0.setter
    def actuator_invweight0(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_invweight0, val_ptr, self.nu*1 * sizeof(c_double))
    
    @property
    def actuator_length0(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_length0, dtype=np.double, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_length0.setter
    def actuator_length0(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_length0, val_ptr, self.nu*1 * sizeof(c_double))
    
    @property
    def actuator_lengthrange(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_lengthrange, dtype=np.double, count=(self.nu*2)), (self.nu, 2, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_lengthrange.setter
    def actuator_lengthrange(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_lengthrange, val_ptr, self.nu*2 * sizeof(c_double))
    
    @property
    def actuator_user(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.actuator_user, dtype=np.double, count=(self.nu*self.nuser_actuator)), (self.nu, self.nuser_actuator, ))
        arr.setflags(write=False)
        return arr
    
    @actuator_user.setter
    def actuator_user(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.actuator_user, val_ptr, self.nu*self.nuser_actuator * sizeof(c_double))
    
    @property
    def numeric_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.numeric_adr, dtype=np.int, count=(self.nnumeric*1)), (self.nnumeric, 1, ))
        arr.setflags(write=False)
        return arr
    
    @numeric_adr.setter
    def numeric_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.numeric_adr, val_ptr, self.nnumeric*1 * sizeof(c_int))
    
    @property
    def numeric_size(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.numeric_size, dtype=np.int, count=(self.nnumeric*1)), (self.nnumeric, 1, ))
        arr.setflags(write=False)
        return arr
    
    @numeric_size.setter
    def numeric_size(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.numeric_size, val_ptr, self.nnumeric*1 * sizeof(c_int))
    
    @property
    def numeric_data(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.numeric_data, dtype=np.double, count=(self.nnumericdata*1)), (self.nnumericdata, 1, ))
        arr.setflags(write=False)
        return arr
    
    @numeric_data.setter
    def numeric_data(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.numeric_data, val_ptr, self.nnumericdata*1 * sizeof(c_double))
    
    @property
    def text_adr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.text_adr, dtype=np.int, count=(self.ntext*1)), (self.ntext, 1, ))
        arr.setflags(write=False)
        return arr
    
    @text_adr.setter
    def text_adr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.text_adr, val_ptr, self.ntext*1 * sizeof(c_int))
    
    @property
    def text_data(self):
        return self._wrapped.contents.text_data
    
    @property
    def name_bodyadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_bodyadr, dtype=np.int, count=(self.nbody*1)), (self.nbody, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_bodyadr.setter
    def name_bodyadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_bodyadr, val_ptr, self.nbody*1 * sizeof(c_int))
    
    @property
    def name_jntadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_jntadr, dtype=np.int, count=(self.njnt*1)), (self.njnt, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_jntadr.setter
    def name_jntadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_jntadr, val_ptr, self.njnt*1 * sizeof(c_int))
    
    @property
    def name_geomadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_geomadr, dtype=np.int, count=(self.ngeom*1)), (self.ngeom, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_geomadr.setter
    def name_geomadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_geomadr, val_ptr, self.ngeom*1 * sizeof(c_int))
    
    @property
    def name_siteadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_siteadr, dtype=np.int, count=(self.nsite*1)), (self.nsite, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_siteadr.setter
    def name_siteadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_siteadr, val_ptr, self.nsite*1 * sizeof(c_int))
    
    @property
    def name_meshadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_meshadr, dtype=np.int, count=(self.nmesh*1)), (self.nmesh, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_meshadr.setter
    def name_meshadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_meshadr, val_ptr, self.nmesh*1 * sizeof(c_int))
    
    @property
    def name_hfieldadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_hfieldadr, dtype=np.int, count=(self.nhfield*1)), (self.nhfield, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_hfieldadr.setter
    def name_hfieldadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_hfieldadr, val_ptr, self.nhfield*1 * sizeof(c_int))
    
    @property
    def name_eqadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_eqadr, dtype=np.int, count=(self.neq*1)), (self.neq, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_eqadr.setter
    def name_eqadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_eqadr, val_ptr, self.neq*1 * sizeof(c_int))
    
    @property
    def name_tendonadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_tendonadr, dtype=np.int, count=(self.ntendon*1)), (self.ntendon, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_tendonadr.setter
    def name_tendonadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_tendonadr, val_ptr, self.ntendon*1 * sizeof(c_int))
    
    @property
    def name_actuatoradr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_actuatoradr, dtype=np.int, count=(self.nu*1)), (self.nu, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_actuatoradr.setter
    def name_actuatoradr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_actuatoradr, val_ptr, self.nu*1 * sizeof(c_int))
    
    @property
    def name_numericadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_numericadr, dtype=np.int, count=(self.nnumeric*1)), (self.nnumeric, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_numericadr.setter
    def name_numericadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_numericadr, val_ptr, self.nnumeric*1 * sizeof(c_int))
    
    @property
    def name_textadr(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.name_textadr, dtype=np.int, count=(self.ntext*1)), (self.ntext, 1, ))
        arr.setflags(write=False)
        return arr
    
    @name_textadr.setter
    def name_textadr(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.name_textadr, val_ptr, self.ntext*1 * sizeof(c_int))
    
    @property
    def names(self):
        return self._wrapped.contents.names
    
    @property
    def key_qpos(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.key_qpos, dtype=np.double, count=(self._size_src.mjNKEY*self.nq)), (self._size_src.mjNKEY, self.nq, ))
        arr.setflags(write=False)
        return arr
    
    @key_qpos.setter
    def key_qpos(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.key_qpos, val_ptr, self._size_src.mjNKEY*self.nq * sizeof(c_double))
    
    @property
    def key_qvel(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.key_qvel, dtype=np.double, count=(self._size_src.mjNKEY*self.nv)), (self._size_src.mjNKEY, self.nv, ))
        arr.setflags(write=False)
        return arr
    
    @key_qvel.setter
    def key_qvel(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.key_qvel, val_ptr, self._size_src.mjNKEY*self.nv * sizeof(c_double))
    
    @property
    def key_act(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.key_act, dtype=np.double, count=(self._size_src.mjNKEY*self.na)), (self._size_src.mjNKEY, self.na, ))
        arr.setflags(write=False)
        return arr
    
    @key_act.setter
    def key_act(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.key_act, val_ptr, self._size_src.mjNKEY*self.na * sizeof(c_double))
    
    @property
    def cam_objtype(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cam_objtype, dtype=np.int, count=(self.ncam*1)), (self.ncam, 1, ))
        arr.setflags(write=False)
        return arr
    
    @cam_objtype.setter
    def cam_objtype(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.cam_objtype, val_ptr, self.ncam*1 * sizeof(c_int))
    
    @property
    def cam_objid(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cam_objid, dtype=np.int, count=(self.ncam*1)), (self.ncam, 1, ))
        arr.setflags(write=False)
        return arr
    
    @cam_objid.setter
    def cam_objid(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.cam_objid, val_ptr, self.ncam*1 * sizeof(c_int))
    
    @property
    def cam_resolution(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cam_resolution, dtype=np.int, count=(self.ncam*2)), (self.ncam, 2, ))
        arr.setflags(write=False)
        return arr
    
    @cam_resolution.setter
    def cam_resolution(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_int))
        memmove(self._wrapped.contents.cam_resolution, val_ptr, self.ncam*2 * sizeof(c_int))
    
    @property
    def cam_fov(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cam_fov, dtype=np.double, count=(self.ncam*2)), (self.ncam, 2, ))
        arr.setflags(write=False)
        return arr
    
    @cam_fov.setter
    def cam_fov(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cam_fov, val_ptr, self.ncam*2 * sizeof(c_double))
    
    @property
    def cam_ipd(self):
        arr = np.reshape(np.fromiter(self._wrapped.contents.cam_ipd, dtype=np.double, count=(self.ncam*1)), (self.ncam, 1, ))
        arr.setflags(write=False)
        return arr
    
    @cam_ipd.setter
    def cam_ipd(self, value):
        val_ptr = np.array(value).ctypes.data_as(POINTER(c_double))
        memmove(self._wrapped.contents.cam_ipd, val_ptr, self.ncam*1 * sizeof(c_double))
