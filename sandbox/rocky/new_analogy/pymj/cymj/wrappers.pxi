# cython: language_level=3
# Automatically generated. Do not modify!

include "mujoco.pxd"
cimport numpy as np
import numpy as np

cdef class PyMjOption(object):
    cdef mjOption* ptr
    cdef np.ndarray _gravity
    cdef np.ndarray _wind
    cdef np.ndarray _magnetic
    cdef np.ndarray _o_solref
    cdef np.ndarray _o_solimp
    cdef void _set(self, mjOption* p):
        self.ptr = p
        self._gravity = _wrap_mjtNum_1d(&p.gravity[0], 3)
        self._wind = _wrap_mjtNum_1d(&p.wind[0], 3)
        self._magnetic = _wrap_mjtNum_1d(&p.magnetic[0], 3)
        self._o_solref = _wrap_mjtNum_1d(&p.o_solref[0], 2)
        self._o_solimp = _wrap_mjtNum_1d(&p.o_solimp[0], 3)
    @property
    def timestep(self): return self.ptr.timestep
    @timestep.setter
    def timestep(self, mjtNum x): self.ptr.timestep = x
    @property
    def apirate(self): return self.ptr.apirate
    @apirate.setter
    def apirate(self, mjtNum x): self.ptr.apirate = x
    @property
    def tolerance(self): return self.ptr.tolerance
    @tolerance.setter
    def tolerance(self, mjtNum x): self.ptr.tolerance = x
    @property
    def impratio(self): return self.ptr.impratio
    @impratio.setter
    def impratio(self, mjtNum x): self.ptr.impratio = x
    @property
    def density(self): return self.ptr.density
    @density.setter
    def density(self, mjtNum x): self.ptr.density = x
    @property
    def viscosity(self): return self.ptr.viscosity
    @viscosity.setter
    def viscosity(self, mjtNum x): self.ptr.viscosity = x
    @property
    def o_margin(self): return self.ptr.o_margin
    @o_margin.setter
    def o_margin(self, mjtNum x): self.ptr.o_margin = x
    @property
    def mpr_tolerance(self): return self.ptr.mpr_tolerance
    @mpr_tolerance.setter
    def mpr_tolerance(self, mjtNum x): self.ptr.mpr_tolerance = x
    @property
    def mpr_iterations(self): return self.ptr.mpr_iterations
    @mpr_iterations.setter
    def mpr_iterations(self, int x): self.ptr.mpr_iterations = x
    @property
    def integrator(self): return self.ptr.integrator
    @integrator.setter
    def integrator(self, int x): self.ptr.integrator = x
    @property
    def collision(self): return self.ptr.collision
    @collision.setter
    def collision(self, int x): self.ptr.collision = x
    @property
    def impedance(self): return self.ptr.impedance
    @impedance.setter
    def impedance(self, int x): self.ptr.impedance = x
    @property
    def reference(self): return self.ptr.reference
    @reference.setter
    def reference(self, int x): self.ptr.reference = x
    @property
    def solver(self): return self.ptr.solver
    @solver.setter
    def solver(self, int x): self.ptr.solver = x
    @property
    def iterations(self): return self.ptr.iterations
    @iterations.setter
    def iterations(self, int x): self.ptr.iterations = x
    @property
    def disableflags(self): return self.ptr.disableflags
    @disableflags.setter
    def disableflags(self, int x): self.ptr.disableflags = x
    @property
    def enableflags(self): return self.ptr.enableflags
    @enableflags.setter
    def enableflags(self, int x): self.ptr.enableflags = x
    @property
    def gravity(self): return self._gravity
    @property
    def wind(self): return self._wind
    @property
    def magnetic(self): return self._magnetic
    @property
    def o_solref(self): return self._o_solref
    @property
    def o_solimp(self): return self._o_solimp

cdef PyMjOption WrapMjOption(mjOption* p):
    cdef PyMjOption o = PyMjOption()
    o._set(p)
    return o

cdef class PyMjStatistic(object):
    cdef mjStatistic* ptr
    cdef np.ndarray _center
    cdef void _set(self, mjStatistic* p):
        self.ptr = p
        self._center = _wrap_mjtNum_1d(&p.center[0], 3)
    @property
    def meanmass(self): return self.ptr.meanmass
    @meanmass.setter
    def meanmass(self, mjtNum x): self.ptr.meanmass = x
    @property
    def meansize(self): return self.ptr.meansize
    @meansize.setter
    def meansize(self, mjtNum x): self.ptr.meansize = x
    @property
    def extent(self): return self.ptr.extent
    @extent.setter
    def extent(self, mjtNum x): self.ptr.extent = x
    @property
    def center(self): return self._center

cdef PyMjStatistic WrapMjStatistic(mjStatistic* p):
    cdef PyMjStatistic o = PyMjStatistic()
    o._set(p)
    return o

cdef class PyMjModel(object):
    cdef mjModel* ptr
    cdef PyMjOption _opt
    cdef PyMjStatistic _stat
    cdef np.ndarray _qpos0
    cdef np.ndarray _qpos_spring
    cdef np.ndarray _body_parentid
    cdef np.ndarray _body_rootid
    cdef np.ndarray _body_weldid
    cdef np.ndarray _body_mocapid
    cdef np.ndarray _body_jntnum
    cdef np.ndarray _body_jntadr
    cdef np.ndarray _body_dofnum
    cdef np.ndarray _body_dofadr
    cdef np.ndarray _body_geomnum
    cdef np.ndarray _body_geomadr
    cdef np.ndarray _body_pos
    cdef np.ndarray _body_quat
    cdef np.ndarray _body_ipos
    cdef np.ndarray _body_iquat
    cdef np.ndarray _body_mass
    cdef np.ndarray _body_subtreemass
    cdef np.ndarray _body_inertia
    cdef np.ndarray _body_invweight0
    cdef np.ndarray _body_user
    cdef np.ndarray _jnt_type
    cdef np.ndarray _jnt_qposadr
    cdef np.ndarray _jnt_dofadr
    cdef np.ndarray _jnt_bodyid
    cdef np.ndarray _jnt_limited
    cdef np.ndarray _jnt_solref
    cdef np.ndarray _jnt_solimp
    cdef np.ndarray _jnt_pos
    cdef np.ndarray _jnt_axis
    cdef np.ndarray _jnt_stiffness
    cdef np.ndarray _jnt_range
    cdef np.ndarray _jnt_margin
    cdef np.ndarray _jnt_user
    cdef np.ndarray _dof_bodyid
    cdef np.ndarray _dof_jntid
    cdef np.ndarray _dof_parentid
    cdef np.ndarray _dof_Madr
    cdef np.ndarray _dof_frictional
    cdef np.ndarray _dof_solref
    cdef np.ndarray _dof_solimp
    cdef np.ndarray _dof_frictionloss
    cdef np.ndarray _dof_armature
    cdef np.ndarray _dof_damping
    cdef np.ndarray _dof_invweight0
    cdef np.ndarray _geom_type
    cdef np.ndarray _geom_contype
    cdef np.ndarray _geom_conaffinity
    cdef np.ndarray _geom_condim
    cdef np.ndarray _geom_bodyid
    cdef np.ndarray _geom_dataid
    cdef np.ndarray _geom_matid
    cdef np.ndarray _geom_group
    cdef np.ndarray _geom_solmix
    cdef np.ndarray _geom_solref
    cdef np.ndarray _geom_solimp
    cdef np.ndarray _geom_size
    cdef np.ndarray _geom_rbound
    cdef np.ndarray _geom_pos
    cdef np.ndarray _geom_quat
    cdef np.ndarray _geom_friction
    cdef np.ndarray _geom_margin
    cdef np.ndarray _geom_gap
    cdef np.ndarray _geom_user
    cdef np.ndarray _geom_rgba
    cdef np.ndarray _site_type
    cdef np.ndarray _site_bodyid
    cdef np.ndarray _site_matid
    cdef np.ndarray _site_group
    cdef np.ndarray _site_size
    cdef np.ndarray _site_pos
    cdef np.ndarray _site_quat
    cdef np.ndarray _site_user
    cdef np.ndarray _site_rgba
    cdef np.ndarray _cam_mode
    cdef np.ndarray _cam_bodyid
    cdef np.ndarray _cam_targetbodyid
    cdef np.ndarray _cam_pos
    cdef np.ndarray _cam_quat
    cdef np.ndarray _cam_poscom0
    cdef np.ndarray _cam_pos0
    cdef np.ndarray _cam_mat0
    cdef np.ndarray _cam_fovy
    cdef np.ndarray _cam_ipd
    cdef np.ndarray _light_mode
    cdef np.ndarray _light_bodyid
    cdef np.ndarray _light_targetbodyid
    cdef np.ndarray _light_directional
    cdef np.ndarray _light_castshadow
    cdef np.ndarray _light_active
    cdef np.ndarray _light_pos
    cdef np.ndarray _light_dir
    cdef np.ndarray _light_poscom0
    cdef np.ndarray _light_pos0
    cdef np.ndarray _light_dir0
    cdef np.ndarray _light_attenuation
    cdef np.ndarray _light_cutoff
    cdef np.ndarray _light_exponent
    cdef np.ndarray _light_ambient
    cdef np.ndarray _light_diffuse
    cdef np.ndarray _light_specular
    cdef np.ndarray _mesh_faceadr
    cdef np.ndarray _mesh_facenum
    cdef np.ndarray _mesh_vertadr
    cdef np.ndarray _mesh_vertnum
    cdef np.ndarray _mesh_graphadr
    cdef np.ndarray _mesh_vert
    cdef np.ndarray _mesh_normal
    cdef np.ndarray _mesh_face
    cdef np.ndarray _mesh_graph
    cdef np.ndarray _hfield_size
    cdef np.ndarray _hfield_nrow
    cdef np.ndarray _hfield_ncol
    cdef np.ndarray _hfield_adr
    cdef np.ndarray _hfield_data
    cdef np.ndarray _tex_type
    cdef np.ndarray _tex_height
    cdef np.ndarray _tex_width
    cdef np.ndarray _tex_adr
    cdef np.ndarray _tex_rgb
    cdef np.ndarray _mat_texid
    cdef np.ndarray _mat_texuniform
    cdef np.ndarray _mat_texrepeat
    cdef np.ndarray _mat_emission
    cdef np.ndarray _mat_specular
    cdef np.ndarray _mat_shininess
    cdef np.ndarray _mat_reflectance
    cdef np.ndarray _mat_rgba
    cdef np.ndarray _pair_dim
    cdef np.ndarray _pair_geom1
    cdef np.ndarray _pair_geom2
    cdef np.ndarray _pair_signature
    cdef np.ndarray _pair_solref
    cdef np.ndarray _pair_solimp
    cdef np.ndarray _pair_margin
    cdef np.ndarray _pair_gap
    cdef np.ndarray _pair_friction
    cdef np.ndarray _exclude_signature
    cdef np.ndarray _eq_type
    cdef np.ndarray _eq_obj1id
    cdef np.ndarray _eq_obj2id
    cdef np.ndarray _eq_active
    cdef np.ndarray _eq_solref
    cdef np.ndarray _eq_solimp
    cdef np.ndarray _eq_data
    cdef np.ndarray _tendon_adr
    cdef np.ndarray _tendon_num
    cdef np.ndarray _tendon_matid
    cdef np.ndarray _tendon_limited
    cdef np.ndarray _tendon_frictional
    cdef np.ndarray _tendon_width
    cdef np.ndarray _tendon_solref_lim
    cdef np.ndarray _tendon_solimp_lim
    cdef np.ndarray _tendon_solref_fri
    cdef np.ndarray _tendon_solimp_fri
    cdef np.ndarray _tendon_range
    cdef np.ndarray _tendon_margin
    cdef np.ndarray _tendon_stiffness
    cdef np.ndarray _tendon_damping
    cdef np.ndarray _tendon_frictionloss
    cdef np.ndarray _tendon_lengthspring
    cdef np.ndarray _tendon_length0
    cdef np.ndarray _tendon_invweight0
    cdef np.ndarray _tendon_user
    cdef np.ndarray _tendon_rgba
    cdef np.ndarray _wrap_type
    cdef np.ndarray _wrap_objid
    cdef np.ndarray _wrap_prm
    cdef np.ndarray _actuator_trntype
    cdef np.ndarray _actuator_dyntype
    cdef np.ndarray _actuator_gaintype
    cdef np.ndarray _actuator_biastype
    cdef np.ndarray _actuator_trnid
    cdef np.ndarray _actuator_ctrllimited
    cdef np.ndarray _actuator_forcelimited
    cdef np.ndarray _actuator_dynprm
    cdef np.ndarray _actuator_gainprm
    cdef np.ndarray _actuator_biasprm
    cdef np.ndarray _actuator_ctrlrange
    cdef np.ndarray _actuator_forcerange
    cdef np.ndarray _actuator_gear
    cdef np.ndarray _actuator_cranklength
    cdef np.ndarray _actuator_invweight0
    cdef np.ndarray _actuator_length0
    cdef np.ndarray _actuator_lengthrange
    cdef np.ndarray _actuator_user
    cdef np.ndarray _sensor_type
    cdef np.ndarray _sensor_datatype
    cdef np.ndarray _sensor_needstage
    cdef np.ndarray _sensor_objtype
    cdef np.ndarray _sensor_objid
    cdef np.ndarray _sensor_dim
    cdef np.ndarray _sensor_adr
    cdef np.ndarray _sensor_noise
    cdef np.ndarray _sensor_user
    cdef np.ndarray _numeric_adr
    cdef np.ndarray _numeric_size
    cdef np.ndarray _numeric_data
    cdef np.ndarray _text_adr
    cdef np.ndarray _text_size
    cdef np.ndarray _text_data
    cdef np.ndarray _tuple_adr
    cdef np.ndarray _tuple_size
    cdef np.ndarray _tuple_objtype
    cdef np.ndarray _tuple_objid
    cdef np.ndarray _tuple_objprm
    cdef np.ndarray _key_time
    cdef np.ndarray _key_qpos
    cdef np.ndarray _key_qvel
    cdef np.ndarray _key_act
    cdef np.ndarray _name_bodyadr
    cdef np.ndarray _name_jntadr
    cdef np.ndarray _name_geomadr
    cdef np.ndarray _name_siteadr
    cdef np.ndarray _name_camadr
    cdef np.ndarray _name_lightadr
    cdef np.ndarray _name_meshadr
    cdef np.ndarray _name_hfieldadr
    cdef np.ndarray _name_texadr
    cdef np.ndarray _name_matadr
    cdef np.ndarray _name_eqadr
    cdef np.ndarray _name_tendonadr
    cdef np.ndarray _name_actuatoradr
    cdef np.ndarray _name_sensoradr
    cdef np.ndarray _name_numericadr
    cdef np.ndarray _name_textadr
    cdef np.ndarray _name_tupleadr
    cdef np.ndarray _names
    cdef void _set(self, mjModel* p):
        self.ptr = p
        self._opt = WrapMjOption(&p.opt)
        self._stat = WrapMjStatistic(&p.stat)
        self._qpos0 = _wrap_mjtNum_1d(p.qpos0, p.nq)
        self._qpos_spring = _wrap_mjtNum_1d(p.qpos_spring, p.nq)
        self._body_parentid = _wrap_int_1d(p.body_parentid, p.nbody)
        self._body_rootid = _wrap_int_1d(p.body_rootid, p.nbody)
        self._body_weldid = _wrap_int_1d(p.body_weldid, p.nbody)
        self._body_mocapid = _wrap_int_1d(p.body_mocapid, p.nbody)
        self._body_jntnum = _wrap_int_1d(p.body_jntnum, p.nbody)
        self._body_jntadr = _wrap_int_1d(p.body_jntadr, p.nbody)
        self._body_dofnum = _wrap_int_1d(p.body_dofnum, p.nbody)
        self._body_dofadr = _wrap_int_1d(p.body_dofadr, p.nbody)
        self._body_geomnum = _wrap_int_1d(p.body_geomnum, p.nbody)
        self._body_geomadr = _wrap_int_1d(p.body_geomadr, p.nbody)
        self._body_pos = _wrap_mjtNum_2d(p.body_pos, p.nbody, 3)
        self._body_quat = _wrap_mjtNum_2d(p.body_quat, p.nbody, 4)
        self._body_ipos = _wrap_mjtNum_2d(p.body_ipos, p.nbody, 3)
        self._body_iquat = _wrap_mjtNum_2d(p.body_iquat, p.nbody, 4)
        self._body_mass = _wrap_mjtNum_1d(p.body_mass, p.nbody)
        self._body_subtreemass = _wrap_mjtNum_1d(p.body_subtreemass, p.nbody)
        self._body_inertia = _wrap_mjtNum_2d(p.body_inertia, p.nbody, 3)
        self._body_invweight0 = _wrap_mjtNum_2d(p.body_invweight0, p.nbody, 2)
        self._body_user = _wrap_mjtNum_2d(p.body_user, p.nbody, p.nuser_body)
        self._jnt_type = _wrap_int_1d(p.jnt_type, p.njnt)
        self._jnt_qposadr = _wrap_int_1d(p.jnt_qposadr, p.njnt)
        self._jnt_dofadr = _wrap_int_1d(p.jnt_dofadr, p.njnt)
        self._jnt_bodyid = _wrap_int_1d(p.jnt_bodyid, p.njnt)
        self._jnt_limited = _wrap_mjtByte_1d(p.jnt_limited, p.njnt)
        self._jnt_solref = _wrap_mjtNum_2d(p.jnt_solref, p.njnt, mjNREF)
        self._jnt_solimp = _wrap_mjtNum_2d(p.jnt_solimp, p.njnt, mjNIMP)
        self._jnt_pos = _wrap_mjtNum_2d(p.jnt_pos, p.njnt, 3)
        self._jnt_axis = _wrap_mjtNum_2d(p.jnt_axis, p.njnt, 3)
        self._jnt_stiffness = _wrap_mjtNum_1d(p.jnt_stiffness, p.njnt)
        self._jnt_range = _wrap_mjtNum_2d(p.jnt_range, p.njnt, 2)
        self._jnt_margin = _wrap_mjtNum_1d(p.jnt_margin, p.njnt)
        self._jnt_user = _wrap_mjtNum_2d(p.jnt_user, p.njnt, p.nuser_jnt)
        self._dof_bodyid = _wrap_int_1d(p.dof_bodyid, p.nv)
        self._dof_jntid = _wrap_int_1d(p.dof_jntid, p.nv)
        self._dof_parentid = _wrap_int_1d(p.dof_parentid, p.nv)
        self._dof_Madr = _wrap_int_1d(p.dof_Madr, p.nv)
        self._dof_frictional = _wrap_mjtByte_1d(p.dof_frictional, p.nv)
        self._dof_solref = _wrap_mjtNum_2d(p.dof_solref, p.nv, mjNREF)
        self._dof_solimp = _wrap_mjtNum_2d(p.dof_solimp, p.nv, mjNIMP)
        self._dof_frictionloss = _wrap_mjtNum_1d(p.dof_frictionloss, p.nv)
        self._dof_armature = _wrap_mjtNum_1d(p.dof_armature, p.nv)
        self._dof_damping = _wrap_mjtNum_1d(p.dof_damping, p.nv)
        self._dof_invweight0 = _wrap_mjtNum_1d(p.dof_invweight0, p.nv)
        self._geom_type = _wrap_int_1d(p.geom_type, p.ngeom)
        self._geom_contype = _wrap_int_1d(p.geom_contype, p.ngeom)
        self._geom_conaffinity = _wrap_int_1d(p.geom_conaffinity, p.ngeom)
        self._geom_condim = _wrap_int_1d(p.geom_condim, p.ngeom)
        self._geom_bodyid = _wrap_int_1d(p.geom_bodyid, p.ngeom)
        self._geom_dataid = _wrap_int_1d(p.geom_dataid, p.ngeom)
        self._geom_matid = _wrap_int_1d(p.geom_matid, p.ngeom)
        self._geom_group = _wrap_int_1d(p.geom_group, p.ngeom)
        self._geom_solmix = _wrap_mjtNum_1d(p.geom_solmix, p.ngeom)
        self._geom_solref = _wrap_mjtNum_2d(p.geom_solref, p.ngeom, mjNREF)
        self._geom_solimp = _wrap_mjtNum_2d(p.geom_solimp, p.ngeom, mjNIMP)
        self._geom_size = _wrap_mjtNum_2d(p.geom_size, p.ngeom, 3)
        self._geom_rbound = _wrap_mjtNum_1d(p.geom_rbound, p.ngeom)
        self._geom_pos = _wrap_mjtNum_2d(p.geom_pos, p.ngeom, 3)
        self._geom_quat = _wrap_mjtNum_2d(p.geom_quat, p.ngeom, 4)
        self._geom_friction = _wrap_mjtNum_2d(p.geom_friction, p.ngeom, 3)
        self._geom_margin = _wrap_mjtNum_1d(p.geom_margin, p.ngeom)
        self._geom_gap = _wrap_mjtNum_1d(p.geom_gap, p.ngeom)
        self._geom_user = _wrap_mjtNum_2d(p.geom_user, p.ngeom, p.nuser_geom)
        self._geom_rgba = _wrap_float_2d(p.geom_rgba, p.ngeom, 4)
        self._site_type = _wrap_int_1d(p.site_type, p.nsite)
        self._site_bodyid = _wrap_int_1d(p.site_bodyid, p.nsite)
        self._site_matid = _wrap_int_1d(p.site_matid, p.nsite)
        self._site_group = _wrap_int_1d(p.site_group, p.nsite)
        self._site_size = _wrap_mjtNum_2d(p.site_size, p.nsite, 3)
        self._site_pos = _wrap_mjtNum_2d(p.site_pos, p.nsite, 3)
        self._site_quat = _wrap_mjtNum_2d(p.site_quat, p.nsite, 4)
        self._site_user = _wrap_mjtNum_2d(p.site_user, p.nsite, p.nuser_site)
        self._site_rgba = _wrap_float_2d(p.site_rgba, p.nsite, 4)
        self._cam_mode = _wrap_int_1d(p.cam_mode, p.ncam)
        self._cam_bodyid = _wrap_int_1d(p.cam_bodyid, p.ncam)
        self._cam_targetbodyid = _wrap_int_1d(p.cam_targetbodyid, p.ncam)
        self._cam_pos = _wrap_mjtNum_2d(p.cam_pos, p.ncam, 3)
        self._cam_quat = _wrap_mjtNum_2d(p.cam_quat, p.ncam, 4)
        self._cam_poscom0 = _wrap_mjtNum_2d(p.cam_poscom0, p.ncam, 3)
        self._cam_pos0 = _wrap_mjtNum_2d(p.cam_pos0, p.ncam, 3)
        self._cam_mat0 = _wrap_mjtNum_2d(p.cam_mat0, p.ncam, 9)
        self._cam_fovy = _wrap_mjtNum_1d(p.cam_fovy, p.ncam)
        self._cam_ipd = _wrap_mjtNum_1d(p.cam_ipd, p.ncam)
        self._light_mode = _wrap_int_1d(p.light_mode, p.nlight)
        self._light_bodyid = _wrap_int_1d(p.light_bodyid, p.nlight)
        self._light_targetbodyid = _wrap_int_1d(p.light_targetbodyid, p.nlight)
        self._light_directional = _wrap_mjtByte_1d(p.light_directional, p.nlight)
        self._light_castshadow = _wrap_mjtByte_1d(p.light_castshadow, p.nlight)
        self._light_active = _wrap_mjtByte_1d(p.light_active, p.nlight)
        self._light_pos = _wrap_mjtNum_2d(p.light_pos, p.nlight, 3)
        self._light_dir = _wrap_mjtNum_2d(p.light_dir, p.nlight, 3)
        self._light_poscom0 = _wrap_mjtNum_2d(p.light_poscom0, p.nlight, 3)
        self._light_pos0 = _wrap_mjtNum_2d(p.light_pos0, p.nlight, 3)
        self._light_dir0 = _wrap_mjtNum_2d(p.light_dir0, p.nlight, 3)
        self._light_attenuation = _wrap_float_2d(p.light_attenuation, p.nlight, 3)
        self._light_cutoff = _wrap_float_1d(p.light_cutoff, p.nlight)
        self._light_exponent = _wrap_float_1d(p.light_exponent, p.nlight)
        self._light_ambient = _wrap_float_2d(p.light_ambient, p.nlight, 3)
        self._light_diffuse = _wrap_float_2d(p.light_diffuse, p.nlight, 3)
        self._light_specular = _wrap_float_2d(p.light_specular, p.nlight, 3)
        self._mesh_faceadr = _wrap_int_1d(p.mesh_faceadr, p.nmesh)
        self._mesh_facenum = _wrap_int_1d(p.mesh_facenum, p.nmesh)
        self._mesh_vertadr = _wrap_int_1d(p.mesh_vertadr, p.nmesh)
        self._mesh_vertnum = _wrap_int_1d(p.mesh_vertnum, p.nmesh)
        self._mesh_graphadr = _wrap_int_1d(p.mesh_graphadr, p.nmesh)
        self._mesh_vert = _wrap_float_2d(p.mesh_vert, p.nmeshvert, 3)
        self._mesh_normal = _wrap_float_2d(p.mesh_normal, p.nmeshvert, 3)
        self._mesh_face = _wrap_int_2d(p.mesh_face, p.nmeshface, 3)
        self._mesh_graph = _wrap_int_1d(p.mesh_graph, p.nmeshgraph)
        self._hfield_size = _wrap_mjtNum_2d(p.hfield_size, p.nhfield, 4)
        self._hfield_nrow = _wrap_int_1d(p.hfield_nrow, p.nhfield)
        self._hfield_ncol = _wrap_int_1d(p.hfield_ncol, p.nhfield)
        self._hfield_adr = _wrap_int_1d(p.hfield_adr, p.nhfield)
        self._hfield_data = _wrap_float_1d(p.hfield_data, p.nhfielddata)
        self._tex_type = _wrap_int_1d(p.tex_type, p.ntex)
        self._tex_height = _wrap_int_1d(p.tex_height, p.ntex)
        self._tex_width = _wrap_int_1d(p.tex_width, p.ntex)
        self._tex_adr = _wrap_int_1d(p.tex_adr, p.ntex)
        self._tex_rgb = _wrap_mjtByte_1d(p.tex_rgb, p.ntexdata)
        self._mat_texid = _wrap_int_1d(p.mat_texid, p.nmat)
        self._mat_texuniform = _wrap_mjtByte_1d(p.mat_texuniform, p.nmat)
        self._mat_texrepeat = _wrap_float_2d(p.mat_texrepeat, p.nmat, 2)
        self._mat_emission = _wrap_float_1d(p.mat_emission, p.nmat)
        self._mat_specular = _wrap_float_1d(p.mat_specular, p.nmat)
        self._mat_shininess = _wrap_float_1d(p.mat_shininess, p.nmat)
        self._mat_reflectance = _wrap_float_1d(p.mat_reflectance, p.nmat)
        self._mat_rgba = _wrap_float_2d(p.mat_rgba, p.nmat, 4)
        self._pair_dim = _wrap_int_1d(p.pair_dim, p.npair)
        self._pair_geom1 = _wrap_int_1d(p.pair_geom1, p.npair)
        self._pair_geom2 = _wrap_int_1d(p.pair_geom2, p.npair)
        self._pair_signature = _wrap_int_1d(p.pair_signature, p.npair)
        self._pair_solref = _wrap_mjtNum_2d(p.pair_solref, p.npair, mjNREF)
        self._pair_solimp = _wrap_mjtNum_2d(p.pair_solimp, p.npair, mjNIMP)
        self._pair_margin = _wrap_mjtNum_1d(p.pair_margin, p.npair)
        self._pair_gap = _wrap_mjtNum_1d(p.pair_gap, p.npair)
        self._pair_friction = _wrap_mjtNum_2d(p.pair_friction, p.npair, 5)
        self._exclude_signature = _wrap_int_1d(p.exclude_signature, p.nexclude)
        self._eq_type = _wrap_int_1d(p.eq_type, p.neq)
        self._eq_obj1id = _wrap_int_1d(p.eq_obj1id, p.neq)
        self._eq_obj2id = _wrap_int_1d(p.eq_obj2id, p.neq)
        self._eq_active = _wrap_mjtByte_1d(p.eq_active, p.neq)
        self._eq_solref = _wrap_mjtNum_2d(p.eq_solref, p.neq, mjNREF)
        self._eq_solimp = _wrap_mjtNum_2d(p.eq_solimp, p.neq, mjNIMP)
        self._eq_data = _wrap_mjtNum_2d(p.eq_data, p.neq, mjNEQDATA)
        self._tendon_adr = _wrap_int_1d(p.tendon_adr, p.ntendon)
        self._tendon_num = _wrap_int_1d(p.tendon_num, p.ntendon)
        self._tendon_matid = _wrap_int_1d(p.tendon_matid, p.ntendon)
        self._tendon_limited = _wrap_mjtByte_1d(p.tendon_limited, p.ntendon)
        self._tendon_frictional = _wrap_mjtByte_1d(p.tendon_frictional, p.ntendon)
        self._tendon_width = _wrap_mjtNum_1d(p.tendon_width, p.ntendon)
        self._tendon_solref_lim = _wrap_mjtNum_2d(p.tendon_solref_lim, p.ntendon, mjNREF)
        self._tendon_solimp_lim = _wrap_mjtNum_2d(p.tendon_solimp_lim, p.ntendon, mjNIMP)
        self._tendon_solref_fri = _wrap_mjtNum_2d(p.tendon_solref_fri, p.ntendon, mjNREF)
        self._tendon_solimp_fri = _wrap_mjtNum_2d(p.tendon_solimp_fri, p.ntendon, mjNIMP)
        self._tendon_range = _wrap_mjtNum_2d(p.tendon_range, p.ntendon, 2)
        self._tendon_margin = _wrap_mjtNum_1d(p.tendon_margin, p.ntendon)
        self._tendon_stiffness = _wrap_mjtNum_1d(p.tendon_stiffness, p.ntendon)
        self._tendon_damping = _wrap_mjtNum_1d(p.tendon_damping, p.ntendon)
        self._tendon_frictionloss = _wrap_mjtNum_1d(p.tendon_frictionloss, p.ntendon)
        self._tendon_lengthspring = _wrap_mjtNum_1d(p.tendon_lengthspring, p.ntendon)
        self._tendon_length0 = _wrap_mjtNum_1d(p.tendon_length0, p.ntendon)
        self._tendon_invweight0 = _wrap_mjtNum_1d(p.tendon_invweight0, p.ntendon)
        self._tendon_user = _wrap_mjtNum_2d(p.tendon_user, p.ntendon, p.nuser_tendon)
        self._tendon_rgba = _wrap_float_2d(p.tendon_rgba, p.ntendon, 4)
        self._wrap_type = _wrap_int_1d(p.wrap_type, p.nwrap)
        self._wrap_objid = _wrap_int_1d(p.wrap_objid, p.nwrap)
        self._wrap_prm = _wrap_mjtNum_1d(p.wrap_prm, p.nwrap)
        self._actuator_trntype = _wrap_int_1d(p.actuator_trntype, p.nu)
        self._actuator_dyntype = _wrap_int_1d(p.actuator_dyntype, p.nu)
        self._actuator_gaintype = _wrap_int_1d(p.actuator_gaintype, p.nu)
        self._actuator_biastype = _wrap_int_1d(p.actuator_biastype, p.nu)
        self._actuator_trnid = _wrap_int_2d(p.actuator_trnid, p.nu, 2)
        self._actuator_ctrllimited = _wrap_mjtByte_1d(p.actuator_ctrllimited, p.nu)
        self._actuator_forcelimited = _wrap_mjtByte_1d(p.actuator_forcelimited, p.nu)
        self._actuator_dynprm = _wrap_mjtNum_2d(p.actuator_dynprm, p.nu, mjNDYN)
        self._actuator_gainprm = _wrap_mjtNum_2d(p.actuator_gainprm, p.nu, mjNGAIN)
        self._actuator_biasprm = _wrap_mjtNum_2d(p.actuator_biasprm, p.nu, mjNBIAS)
        self._actuator_ctrlrange = _wrap_mjtNum_2d(p.actuator_ctrlrange, p.nu, 2)
        self._actuator_forcerange = _wrap_mjtNum_2d(p.actuator_forcerange, p.nu, 2)
        self._actuator_gear = _wrap_mjtNum_2d(p.actuator_gear, p.nu, 6)
        self._actuator_cranklength = _wrap_mjtNum_1d(p.actuator_cranklength, p.nu)
        self._actuator_invweight0 = _wrap_mjtNum_1d(p.actuator_invweight0, p.nu)
        self._actuator_length0 = _wrap_mjtNum_1d(p.actuator_length0, p.nu)
        self._actuator_lengthrange = _wrap_mjtNum_2d(p.actuator_lengthrange, p.nu, 2)
        self._actuator_user = _wrap_mjtNum_2d(p.actuator_user, p.nu, p.nuser_actuator)
        self._sensor_type = _wrap_int_1d(p.sensor_type, p.nsensor)
        self._sensor_datatype = _wrap_int_1d(p.sensor_datatype, p.nsensor)
        self._sensor_needstage = _wrap_int_1d(p.sensor_needstage, p.nsensor)
        self._sensor_objtype = _wrap_int_1d(p.sensor_objtype, p.nsensor)
        self._sensor_objid = _wrap_int_1d(p.sensor_objid, p.nsensor)
        self._sensor_dim = _wrap_int_1d(p.sensor_dim, p.nsensor)
        self._sensor_adr = _wrap_int_1d(p.sensor_adr, p.nsensor)
        self._sensor_noise = _wrap_mjtNum_1d(p.sensor_noise, p.nsensor)
        self._sensor_user = _wrap_mjtNum_2d(p.sensor_user, p.nsensor, p.nuser_sensor)
        self._numeric_adr = _wrap_int_1d(p.numeric_adr, p.nnumeric)
        self._numeric_size = _wrap_int_1d(p.numeric_size, p.nnumeric)
        self._numeric_data = _wrap_mjtNum_1d(p.numeric_data, p.nnumericdata)
        self._text_adr = _wrap_int_1d(p.text_adr, p.ntext)
        self._text_size = _wrap_int_1d(p.text_size, p.ntext)
        self._text_data = _wrap_char_1d(p.text_data, p.ntextdata)
        self._tuple_adr = _wrap_int_1d(p.tuple_adr, p.ntuple)
        self._tuple_size = _wrap_int_1d(p.tuple_size, p.ntuple)
        self._tuple_objtype = _wrap_int_1d(p.tuple_objtype, p.ntupledata)
        self._tuple_objid = _wrap_int_1d(p.tuple_objid, p.ntupledata)
        self._tuple_objprm = _wrap_mjtNum_1d(p.tuple_objprm, p.ntupledata)
        self._key_time = _wrap_mjtNum_1d(p.key_time, p.nkey)
        self._key_qpos = _wrap_mjtNum_2d(p.key_qpos, p.nkey, p.nq)
        self._key_qvel = _wrap_mjtNum_2d(p.key_qvel, p.nkey, p.nv)
        self._key_act = _wrap_mjtNum_2d(p.key_act, p.nkey, p.na)
        self._name_bodyadr = _wrap_int_1d(p.name_bodyadr, p.nbody)
        self._name_jntadr = _wrap_int_1d(p.name_jntadr, p.njnt)
        self._name_geomadr = _wrap_int_1d(p.name_geomadr, p.ngeom)
        self._name_siteadr = _wrap_int_1d(p.name_siteadr, p.nsite)
        self._name_camadr = _wrap_int_1d(p.name_camadr, p.ncam)
        self._name_lightadr = _wrap_int_1d(p.name_lightadr, p.nlight)
        self._name_meshadr = _wrap_int_1d(p.name_meshadr, p.nmesh)
        self._name_hfieldadr = _wrap_int_1d(p.name_hfieldadr, p.nhfield)
        self._name_texadr = _wrap_int_1d(p.name_texadr, p.ntex)
        self._name_matadr = _wrap_int_1d(p.name_matadr, p.nmat)
        self._name_eqadr = _wrap_int_1d(p.name_eqadr, p.neq)
        self._name_tendonadr = _wrap_int_1d(p.name_tendonadr, p.ntendon)
        self._name_actuatoradr = _wrap_int_1d(p.name_actuatoradr, p.nu)
        self._name_sensoradr = _wrap_int_1d(p.name_sensoradr, p.nsensor)
        self._name_numericadr = _wrap_int_1d(p.name_numericadr, p.nnumeric)
        self._name_textadr = _wrap_int_1d(p.name_textadr, p.ntext)
        self._name_tupleadr = _wrap_int_1d(p.name_tupleadr, p.ntuple)
        self._names = _wrap_char_1d(p.names, p.nnames)
    @property
    def nq(self): return self.ptr.nq
    @nq.setter
    def nq(self, int x): self.ptr.nq = x
    @property
    def nv(self): return self.ptr.nv
    @nv.setter
    def nv(self, int x): self.ptr.nv = x
    @property
    def nu(self): return self.ptr.nu
    @nu.setter
    def nu(self, int x): self.ptr.nu = x
    @property
    def na(self): return self.ptr.na
    @na.setter
    def na(self, int x): self.ptr.na = x
    @property
    def nbody(self): return self.ptr.nbody
    @nbody.setter
    def nbody(self, int x): self.ptr.nbody = x
    @property
    def njnt(self): return self.ptr.njnt
    @njnt.setter
    def njnt(self, int x): self.ptr.njnt = x
    @property
    def ngeom(self): return self.ptr.ngeom
    @ngeom.setter
    def ngeom(self, int x): self.ptr.ngeom = x
    @property
    def nsite(self): return self.ptr.nsite
    @nsite.setter
    def nsite(self, int x): self.ptr.nsite = x
    @property
    def ncam(self): return self.ptr.ncam
    @ncam.setter
    def ncam(self, int x): self.ptr.ncam = x
    @property
    def nlight(self): return self.ptr.nlight
    @nlight.setter
    def nlight(self, int x): self.ptr.nlight = x
    @property
    def nmesh(self): return self.ptr.nmesh
    @nmesh.setter
    def nmesh(self, int x): self.ptr.nmesh = x
    @property
    def nmeshvert(self): return self.ptr.nmeshvert
    @nmeshvert.setter
    def nmeshvert(self, int x): self.ptr.nmeshvert = x
    @property
    def nmeshface(self): return self.ptr.nmeshface
    @nmeshface.setter
    def nmeshface(self, int x): self.ptr.nmeshface = x
    @property
    def nmeshgraph(self): return self.ptr.nmeshgraph
    @nmeshgraph.setter
    def nmeshgraph(self, int x): self.ptr.nmeshgraph = x
    @property
    def nhfield(self): return self.ptr.nhfield
    @nhfield.setter
    def nhfield(self, int x): self.ptr.nhfield = x
    @property
    def nhfielddata(self): return self.ptr.nhfielddata
    @nhfielddata.setter
    def nhfielddata(self, int x): self.ptr.nhfielddata = x
    @property
    def ntex(self): return self.ptr.ntex
    @ntex.setter
    def ntex(self, int x): self.ptr.ntex = x
    @property
    def ntexdata(self): return self.ptr.ntexdata
    @ntexdata.setter
    def ntexdata(self, int x): self.ptr.ntexdata = x
    @property
    def nmat(self): return self.ptr.nmat
    @nmat.setter
    def nmat(self, int x): self.ptr.nmat = x
    @property
    def npair(self): return self.ptr.npair
    @npair.setter
    def npair(self, int x): self.ptr.npair = x
    @property
    def nexclude(self): return self.ptr.nexclude
    @nexclude.setter
    def nexclude(self, int x): self.ptr.nexclude = x
    @property
    def neq(self): return self.ptr.neq
    @neq.setter
    def neq(self, int x): self.ptr.neq = x
    @property
    def ntendon(self): return self.ptr.ntendon
    @ntendon.setter
    def ntendon(self, int x): self.ptr.ntendon = x
    @property
    def nwrap(self): return self.ptr.nwrap
    @nwrap.setter
    def nwrap(self, int x): self.ptr.nwrap = x
    @property
    def nsensor(self): return self.ptr.nsensor
    @nsensor.setter
    def nsensor(self, int x): self.ptr.nsensor = x
    @property
    def nnumeric(self): return self.ptr.nnumeric
    @nnumeric.setter
    def nnumeric(self, int x): self.ptr.nnumeric = x
    @property
    def nnumericdata(self): return self.ptr.nnumericdata
    @nnumericdata.setter
    def nnumericdata(self, int x): self.ptr.nnumericdata = x
    @property
    def ntext(self): return self.ptr.ntext
    @ntext.setter
    def ntext(self, int x): self.ptr.ntext = x
    @property
    def ntextdata(self): return self.ptr.ntextdata
    @ntextdata.setter
    def ntextdata(self, int x): self.ptr.ntextdata = x
    @property
    def ntuple(self): return self.ptr.ntuple
    @ntuple.setter
    def ntuple(self, int x): self.ptr.ntuple = x
    @property
    def ntupledata(self): return self.ptr.ntupledata
    @ntupledata.setter
    def ntupledata(self, int x): self.ptr.ntupledata = x
    @property
    def nkey(self): return self.ptr.nkey
    @nkey.setter
    def nkey(self, int x): self.ptr.nkey = x
    @property
    def nuser_body(self): return self.ptr.nuser_body
    @nuser_body.setter
    def nuser_body(self, int x): self.ptr.nuser_body = x
    @property
    def nuser_jnt(self): return self.ptr.nuser_jnt
    @nuser_jnt.setter
    def nuser_jnt(self, int x): self.ptr.nuser_jnt = x
    @property
    def nuser_geom(self): return self.ptr.nuser_geom
    @nuser_geom.setter
    def nuser_geom(self, int x): self.ptr.nuser_geom = x
    @property
    def nuser_site(self): return self.ptr.nuser_site
    @nuser_site.setter
    def nuser_site(self, int x): self.ptr.nuser_site = x
    @property
    def nuser_tendon(self): return self.ptr.nuser_tendon
    @nuser_tendon.setter
    def nuser_tendon(self, int x): self.ptr.nuser_tendon = x
    @property
    def nuser_actuator(self): return self.ptr.nuser_actuator
    @nuser_actuator.setter
    def nuser_actuator(self, int x): self.ptr.nuser_actuator = x
    @property
    def nuser_sensor(self): return self.ptr.nuser_sensor
    @nuser_sensor.setter
    def nuser_sensor(self, int x): self.ptr.nuser_sensor = x
    @property
    def nnames(self): return self.ptr.nnames
    @nnames.setter
    def nnames(self, int x): self.ptr.nnames = x
    @property
    def nM(self): return self.ptr.nM
    @nM.setter
    def nM(self, int x): self.ptr.nM = x
    @property
    def nemax(self): return self.ptr.nemax
    @nemax.setter
    def nemax(self, int x): self.ptr.nemax = x
    @property
    def njmax(self): return self.ptr.njmax
    @njmax.setter
    def njmax(self, int x): self.ptr.njmax = x
    @property
    def nconmax(self): return self.ptr.nconmax
    @nconmax.setter
    def nconmax(self, int x): self.ptr.nconmax = x
    @property
    def nstack(self): return self.ptr.nstack
    @nstack.setter
    def nstack(self, int x): self.ptr.nstack = x
    @property
    def nuserdata(self): return self.ptr.nuserdata
    @nuserdata.setter
    def nuserdata(self, int x): self.ptr.nuserdata = x
    @property
    def nmocap(self): return self.ptr.nmocap
    @nmocap.setter
    def nmocap(self, int x): self.ptr.nmocap = x
    @property
    def nsensordata(self): return self.ptr.nsensordata
    @nsensordata.setter
    def nsensordata(self, int x): self.ptr.nsensordata = x
    @property
    def nbuffer(self): return self.ptr.nbuffer
    @nbuffer.setter
    def nbuffer(self, int x): self.ptr.nbuffer = x
    @property
    def opt(self): return self._opt
    @property
    def stat(self): return self._stat
    @property
    def qpos0(self): return self._qpos0
    @property
    def qpos_spring(self): return self._qpos_spring
    @property
    def body_parentid(self): return self._body_parentid
    @property
    def body_rootid(self): return self._body_rootid
    @property
    def body_weldid(self): return self._body_weldid
    @property
    def body_mocapid(self): return self._body_mocapid
    @property
    def body_jntnum(self): return self._body_jntnum
    @property
    def body_jntadr(self): return self._body_jntadr
    @property
    def body_dofnum(self): return self._body_dofnum
    @property
    def body_dofadr(self): return self._body_dofadr
    @property
    def body_geomnum(self): return self._body_geomnum
    @property
    def body_geomadr(self): return self._body_geomadr
    @property
    def body_pos(self): return self._body_pos
    @property
    def body_quat(self): return self._body_quat
    @property
    def body_ipos(self): return self._body_ipos
    @property
    def body_iquat(self): return self._body_iquat
    @property
    def body_mass(self): return self._body_mass
    @property
    def body_subtreemass(self): return self._body_subtreemass
    @property
    def body_inertia(self): return self._body_inertia
    @property
    def body_invweight0(self): return self._body_invweight0
    @property
    def body_user(self): return self._body_user
    @property
    def jnt_type(self): return self._jnt_type
    @property
    def jnt_qposadr(self): return self._jnt_qposadr
    @property
    def jnt_dofadr(self): return self._jnt_dofadr
    @property
    def jnt_bodyid(self): return self._jnt_bodyid
    @property
    def jnt_limited(self): return self._jnt_limited
    @property
    def jnt_solref(self): return self._jnt_solref
    @property
    def jnt_solimp(self): return self._jnt_solimp
    @property
    def jnt_pos(self): return self._jnt_pos
    @property
    def jnt_axis(self): return self._jnt_axis
    @property
    def jnt_stiffness(self): return self._jnt_stiffness
    @property
    def jnt_range(self): return self._jnt_range
    @property
    def jnt_margin(self): return self._jnt_margin
    @property
    def jnt_user(self): return self._jnt_user
    @property
    def dof_bodyid(self): return self._dof_bodyid
    @property
    def dof_jntid(self): return self._dof_jntid
    @property
    def dof_parentid(self): return self._dof_parentid
    @property
    def dof_Madr(self): return self._dof_Madr
    @property
    def dof_frictional(self): return self._dof_frictional
    @property
    def dof_solref(self): return self._dof_solref
    @property
    def dof_solimp(self): return self._dof_solimp
    @property
    def dof_frictionloss(self): return self._dof_frictionloss
    @property
    def dof_armature(self): return self._dof_armature
    @property
    def dof_damping(self): return self._dof_damping
    @property
    def dof_invweight0(self): return self._dof_invweight0
    @property
    def geom_type(self): return self._geom_type
    @property
    def geom_contype(self): return self._geom_contype
    @property
    def geom_conaffinity(self): return self._geom_conaffinity
    @property
    def geom_condim(self): return self._geom_condim
    @property
    def geom_bodyid(self): return self._geom_bodyid
    @property
    def geom_dataid(self): return self._geom_dataid
    @property
    def geom_matid(self): return self._geom_matid
    @property
    def geom_group(self): return self._geom_group
    @property
    def geom_solmix(self): return self._geom_solmix
    @property
    def geom_solref(self): return self._geom_solref
    @property
    def geom_solimp(self): return self._geom_solimp
    @property
    def geom_size(self): return self._geom_size
    @property
    def geom_rbound(self): return self._geom_rbound
    @property
    def geom_pos(self): return self._geom_pos
    @property
    def geom_quat(self): return self._geom_quat
    @property
    def geom_friction(self): return self._geom_friction
    @property
    def geom_margin(self): return self._geom_margin
    @property
    def geom_gap(self): return self._geom_gap
    @property
    def geom_user(self): return self._geom_user
    @property
    def geom_rgba(self): return self._geom_rgba
    @property
    def site_type(self): return self._site_type
    @property
    def site_bodyid(self): return self._site_bodyid
    @property
    def site_matid(self): return self._site_matid
    @property
    def site_group(self): return self._site_group
    @property
    def site_size(self): return self._site_size
    @property
    def site_pos(self): return self._site_pos
    @property
    def site_quat(self): return self._site_quat
    @property
    def site_user(self): return self._site_user
    @property
    def site_rgba(self): return self._site_rgba
    @property
    def cam_mode(self): return self._cam_mode
    @property
    def cam_bodyid(self): return self._cam_bodyid
    @property
    def cam_targetbodyid(self): return self._cam_targetbodyid
    @property
    def cam_pos(self): return self._cam_pos
    @property
    def cam_quat(self): return self._cam_quat
    @property
    def cam_poscom0(self): return self._cam_poscom0
    @property
    def cam_pos0(self): return self._cam_pos0
    @property
    def cam_mat0(self): return self._cam_mat0
    @property
    def cam_fovy(self): return self._cam_fovy
    @property
    def cam_ipd(self): return self._cam_ipd
    @property
    def light_mode(self): return self._light_mode
    @property
    def light_bodyid(self): return self._light_bodyid
    @property
    def light_targetbodyid(self): return self._light_targetbodyid
    @property
    def light_directional(self): return self._light_directional
    @property
    def light_castshadow(self): return self._light_castshadow
    @property
    def light_active(self): return self._light_active
    @property
    def light_pos(self): return self._light_pos
    @property
    def light_dir(self): return self._light_dir
    @property
    def light_poscom0(self): return self._light_poscom0
    @property
    def light_pos0(self): return self._light_pos0
    @property
    def light_dir0(self): return self._light_dir0
    @property
    def light_attenuation(self): return self._light_attenuation
    @property
    def light_cutoff(self): return self._light_cutoff
    @property
    def light_exponent(self): return self._light_exponent
    @property
    def light_ambient(self): return self._light_ambient
    @property
    def light_diffuse(self): return self._light_diffuse
    @property
    def light_specular(self): return self._light_specular
    @property
    def mesh_faceadr(self): return self._mesh_faceadr
    @property
    def mesh_facenum(self): return self._mesh_facenum
    @property
    def mesh_vertadr(self): return self._mesh_vertadr
    @property
    def mesh_vertnum(self): return self._mesh_vertnum
    @property
    def mesh_graphadr(self): return self._mesh_graphadr
    @property
    def mesh_vert(self): return self._mesh_vert
    @property
    def mesh_normal(self): return self._mesh_normal
    @property
    def mesh_face(self): return self._mesh_face
    @property
    def mesh_graph(self): return self._mesh_graph
    @property
    def hfield_size(self): return self._hfield_size
    @property
    def hfield_nrow(self): return self._hfield_nrow
    @property
    def hfield_ncol(self): return self._hfield_ncol
    @property
    def hfield_adr(self): return self._hfield_adr
    @property
    def hfield_data(self): return self._hfield_data
    @property
    def tex_type(self): return self._tex_type
    @property
    def tex_height(self): return self._tex_height
    @property
    def tex_width(self): return self._tex_width
    @property
    def tex_adr(self): return self._tex_adr
    @property
    def tex_rgb(self): return self._tex_rgb
    @property
    def mat_texid(self): return self._mat_texid
    @property
    def mat_texuniform(self): return self._mat_texuniform
    @property
    def mat_texrepeat(self): return self._mat_texrepeat
    @property
    def mat_emission(self): return self._mat_emission
    @property
    def mat_specular(self): return self._mat_specular
    @property
    def mat_shininess(self): return self._mat_shininess
    @property
    def mat_reflectance(self): return self._mat_reflectance
    @property
    def mat_rgba(self): return self._mat_rgba
    @property
    def pair_dim(self): return self._pair_dim
    @property
    def pair_geom1(self): return self._pair_geom1
    @property
    def pair_geom2(self): return self._pair_geom2
    @property
    def pair_signature(self): return self._pair_signature
    @property
    def pair_solref(self): return self._pair_solref
    @property
    def pair_solimp(self): return self._pair_solimp
    @property
    def pair_margin(self): return self._pair_margin
    @property
    def pair_gap(self): return self._pair_gap
    @property
    def pair_friction(self): return self._pair_friction
    @property
    def exclude_signature(self): return self._exclude_signature
    @property
    def eq_type(self): return self._eq_type
    @property
    def eq_obj1id(self): return self._eq_obj1id
    @property
    def eq_obj2id(self): return self._eq_obj2id
    @property
    def eq_active(self): return self._eq_active
    @property
    def eq_solref(self): return self._eq_solref
    @property
    def eq_solimp(self): return self._eq_solimp
    @property
    def eq_data(self): return self._eq_data
    @property
    def tendon_adr(self): return self._tendon_adr
    @property
    def tendon_num(self): return self._tendon_num
    @property
    def tendon_matid(self): return self._tendon_matid
    @property
    def tendon_limited(self): return self._tendon_limited
    @property
    def tendon_frictional(self): return self._tendon_frictional
    @property
    def tendon_width(self): return self._tendon_width
    @property
    def tendon_solref_lim(self): return self._tendon_solref_lim
    @property
    def tendon_solimp_lim(self): return self._tendon_solimp_lim
    @property
    def tendon_solref_fri(self): return self._tendon_solref_fri
    @property
    def tendon_solimp_fri(self): return self._tendon_solimp_fri
    @property
    def tendon_range(self): return self._tendon_range
    @property
    def tendon_margin(self): return self._tendon_margin
    @property
    def tendon_stiffness(self): return self._tendon_stiffness
    @property
    def tendon_damping(self): return self._tendon_damping
    @property
    def tendon_frictionloss(self): return self._tendon_frictionloss
    @property
    def tendon_lengthspring(self): return self._tendon_lengthspring
    @property
    def tendon_length0(self): return self._tendon_length0
    @property
    def tendon_invweight0(self): return self._tendon_invweight0
    @property
    def tendon_user(self): return self._tendon_user
    @property
    def tendon_rgba(self): return self._tendon_rgba
    @property
    def wrap_type(self): return self._wrap_type
    @property
    def wrap_objid(self): return self._wrap_objid
    @property
    def wrap_prm(self): return self._wrap_prm
    @property
    def actuator_trntype(self): return self._actuator_trntype
    @property
    def actuator_dyntype(self): return self._actuator_dyntype
    @property
    def actuator_gaintype(self): return self._actuator_gaintype
    @property
    def actuator_biastype(self): return self._actuator_biastype
    @property
    def actuator_trnid(self): return self._actuator_trnid
    @property
    def actuator_ctrllimited(self): return self._actuator_ctrllimited
    @property
    def actuator_forcelimited(self): return self._actuator_forcelimited
    @property
    def actuator_dynprm(self): return self._actuator_dynprm
    @property
    def actuator_gainprm(self): return self._actuator_gainprm
    @property
    def actuator_biasprm(self): return self._actuator_biasprm
    @property
    def actuator_ctrlrange(self): return self._actuator_ctrlrange
    @property
    def actuator_forcerange(self): return self._actuator_forcerange
    @property
    def actuator_gear(self): return self._actuator_gear
    @property
    def actuator_cranklength(self): return self._actuator_cranklength
    @property
    def actuator_invweight0(self): return self._actuator_invweight0
    @property
    def actuator_length0(self): return self._actuator_length0
    @property
    def actuator_lengthrange(self): return self._actuator_lengthrange
    @property
    def actuator_user(self): return self._actuator_user
    @property
    def sensor_type(self): return self._sensor_type
    @property
    def sensor_datatype(self): return self._sensor_datatype
    @property
    def sensor_needstage(self): return self._sensor_needstage
    @property
    def sensor_objtype(self): return self._sensor_objtype
    @property
    def sensor_objid(self): return self._sensor_objid
    @property
    def sensor_dim(self): return self._sensor_dim
    @property
    def sensor_adr(self): return self._sensor_adr
    @property
    def sensor_noise(self): return self._sensor_noise
    @property
    def sensor_user(self): return self._sensor_user
    @property
    def numeric_adr(self): return self._numeric_adr
    @property
    def numeric_size(self): return self._numeric_size
    @property
    def numeric_data(self): return self._numeric_data
    @property
    def text_adr(self): return self._text_adr
    @property
    def text_size(self): return self._text_size
    @property
    def text_data(self): return self._text_data
    @property
    def tuple_adr(self): return self._tuple_adr
    @property
    def tuple_size(self): return self._tuple_size
    @property
    def tuple_objtype(self): return self._tuple_objtype
    @property
    def tuple_objid(self): return self._tuple_objid
    @property
    def tuple_objprm(self): return self._tuple_objprm
    @property
    def key_time(self): return self._key_time
    @property
    def key_qpos(self): return self._key_qpos
    @property
    def key_qvel(self): return self._key_qvel
    @property
    def key_act(self): return self._key_act
    @property
    def name_bodyadr(self): return self._name_bodyadr
    @property
    def name_jntadr(self): return self._name_jntadr
    @property
    def name_geomadr(self): return self._name_geomadr
    @property
    def name_siteadr(self): return self._name_siteadr
    @property
    def name_camadr(self): return self._name_camadr
    @property
    def name_lightadr(self): return self._name_lightadr
    @property
    def name_meshadr(self): return self._name_meshadr
    @property
    def name_hfieldadr(self): return self._name_hfieldadr
    @property
    def name_texadr(self): return self._name_texadr
    @property
    def name_matadr(self): return self._name_matadr
    @property
    def name_eqadr(self): return self._name_eqadr
    @property
    def name_tendonadr(self): return self._name_tendonadr
    @property
    def name_actuatoradr(self): return self._name_actuatoradr
    @property
    def name_sensoradr(self): return self._name_sensoradr
    @property
    def name_numericadr(self): return self._name_numericadr
    @property
    def name_textadr(self): return self._name_textadr
    @property
    def name_tupleadr(self): return self._name_tupleadr
    @property
    def names(self): return self._names

cdef PyMjModel WrapMjModel(mjModel* p):
    cdef PyMjModel o = PyMjModel()
    o._set(p)
    return o

cdef class PyMjContact(object):
    cdef mjContact* ptr
    cdef np.ndarray _pos
    cdef np.ndarray _frame
    cdef np.ndarray _friction
    cdef np.ndarray _solref
    cdef np.ndarray _solimp
    cdef np.ndarray _coef
    cdef void _set(self, mjContact* p):
        self.ptr = p
        self._pos = _wrap_mjtNum_1d(&p.pos[0], 3)
        self._frame = _wrap_mjtNum_1d(&p.frame[0], 9)
        self._friction = _wrap_mjtNum_1d(&p.friction[0], 5)
        self._solref = _wrap_mjtNum_1d(&p.solref[0], 2)
        self._solimp = _wrap_mjtNum_1d(&p.solimp[0], 3)
        self._coef = _wrap_mjtNum_1d(&p.coef[0], 5)
    @property
    def dist(self): return self.ptr.dist
    @dist.setter
    def dist(self, mjtNum x): self.ptr.dist = x
    @property
    def includemargin(self): return self.ptr.includemargin
    @includemargin.setter
    def includemargin(self, mjtNum x): self.ptr.includemargin = x
    @property
    def mu(self): return self.ptr.mu
    @mu.setter
    def mu(self, mjtNum x): self.ptr.mu = x
    @property
    def zone(self): return self.ptr.zone
    @zone.setter
    def zone(self, int x): self.ptr.zone = x
    @property
    def dim(self): return self.ptr.dim
    @dim.setter
    def dim(self, int x): self.ptr.dim = x
    @property
    def geom1(self): return self.ptr.geom1
    @geom1.setter
    def geom1(self, int x): self.ptr.geom1 = x
    @property
    def geom2(self): return self.ptr.geom2
    @geom2.setter
    def geom2(self, int x): self.ptr.geom2 = x
    @property
    def exclude(self): return self.ptr.exclude
    @exclude.setter
    def exclude(self, int x): self.ptr.exclude = x
    @property
    def efc_address(self): return self.ptr.efc_address
    @efc_address.setter
    def efc_address(self, int x): self.ptr.efc_address = x
    @property
    def pos(self): return self._pos
    @property
    def frame(self): return self._frame
    @property
    def friction(self): return self._friction
    @property
    def solref(self): return self._solref
    @property
    def solimp(self): return self._solimp
    @property
    def coef(self): return self._coef

cdef PyMjContact WrapMjContact(mjContact* p):
    cdef PyMjContact o = PyMjContact()
    o._set(p)
    return o

cdef class PyMjData(object):
    cdef mjData* ptr
    cdef mjModel* _model
    cdef np.ndarray _qpos
    cdef np.ndarray _qvel
    cdef np.ndarray _act
    cdef np.ndarray _ctrl
    cdef np.ndarray _qfrc_applied
    cdef np.ndarray _xfrc_applied
    cdef np.ndarray _qacc
    cdef np.ndarray _act_dot
    cdef np.ndarray _mocap_pos
    cdef np.ndarray _mocap_quat
    cdef np.ndarray _userdata
    cdef np.ndarray _sensordata
    cdef np.ndarray _xpos
    cdef np.ndarray _xquat
    cdef np.ndarray _xmat
    cdef np.ndarray _xipos
    cdef np.ndarray _ximat
    cdef np.ndarray _xanchor
    cdef np.ndarray _xaxis
    cdef np.ndarray _geom_xpos
    cdef np.ndarray _geom_xmat
    cdef np.ndarray _site_xpos
    cdef np.ndarray _site_xmat
    cdef np.ndarray _cam_xpos
    cdef np.ndarray _cam_xmat
    cdef np.ndarray _light_xpos
    cdef np.ndarray _light_xdir
    cdef np.ndarray _subtree_com
    cdef np.ndarray _cdof
    cdef np.ndarray _cinert
    cdef np.ndarray _ten_wrapadr
    cdef np.ndarray _ten_wrapnum
    cdef np.ndarray _ten_length
    cdef np.ndarray _ten_moment
    cdef np.ndarray _wrap_obj
    cdef np.ndarray _wrap_xpos
    cdef np.ndarray _actuator_length
    cdef np.ndarray _actuator_moment
    cdef np.ndarray _crb
    cdef np.ndarray _qM
    cdef np.ndarray _qLD
    cdef np.ndarray _qLDiagInv
    cdef np.ndarray _qLDiagSqrtInv
    cdef tuple _contact
    cdef np.ndarray _efc_type
    cdef np.ndarray _efc_id
    cdef np.ndarray _efc_rownnz
    cdef np.ndarray _efc_rowadr
    cdef np.ndarray _efc_colind
    cdef np.ndarray _efc_rownnz_T
    cdef np.ndarray _efc_rowadr_T
    cdef np.ndarray _efc_colind_T
    cdef np.ndarray _efc_solref
    cdef np.ndarray _efc_solimp
    cdef np.ndarray _efc_margin
    cdef np.ndarray _efc_frictionloss
    cdef np.ndarray _efc_pos
    cdef np.ndarray _efc_J
    cdef np.ndarray _efc_J_T
    cdef np.ndarray _efc_diagApprox
    cdef np.ndarray _efc_D
    cdef np.ndarray _efc_R
    cdef np.ndarray _efc_AR
    cdef np.ndarray _e_ARchol
    cdef np.ndarray _fc_e_rect
    cdef np.ndarray _fc_AR
    cdef np.ndarray _ten_velocity
    cdef np.ndarray _actuator_velocity
    cdef np.ndarray _cvel
    cdef np.ndarray _cdof_dot
    cdef np.ndarray _qfrc_bias
    cdef np.ndarray _qfrc_passive
    cdef np.ndarray _efc_vel
    cdef np.ndarray _efc_aref
    cdef np.ndarray _subtree_linvel
    cdef np.ndarray _subtree_angmom
    cdef np.ndarray _actuator_force
    cdef np.ndarray _qfrc_actuator
    cdef np.ndarray _qfrc_unc
    cdef np.ndarray _qacc_unc
    cdef np.ndarray _efc_b
    cdef np.ndarray _fc_b
    cdef np.ndarray _efc_force
    cdef np.ndarray _qfrc_constraint
    cdef np.ndarray _qfrc_inverse
    cdef np.ndarray _cacc
    cdef np.ndarray _cfrc_int
    cdef np.ndarray _cfrc_ext
    cdef np.ndarray _nwarning
    cdef np.ndarray _warning_info
    cdef np.ndarray _timer_ncall
    cdef np.ndarray _timer_duration
    cdef np.ndarray _solver_trace
    cdef np.ndarray _solver_fwdinv
    cdef np.ndarray _energy
    cdef void _set(self, mjData* p, mjModel* model):
        self.ptr = p
        self._model = model
        self._qpos = _wrap_mjtNum_1d(p.qpos, model.nq)
        self._qvel = _wrap_mjtNum_1d(p.qvel, model.nv)
        self._act = _wrap_mjtNum_1d(p.act, model.na)
        self._ctrl = _wrap_mjtNum_1d(p.ctrl, model.nu)
        self._qfrc_applied = _wrap_mjtNum_1d(p.qfrc_applied, model.nv)
        self._xfrc_applied = _wrap_mjtNum_2d(p.xfrc_applied, model.nbody, 6)
        self._qacc = _wrap_mjtNum_1d(p.qacc, model.nv)
        self._act_dot = _wrap_mjtNum_1d(p.act_dot, model.na)
        self._mocap_pos = _wrap_mjtNum_2d(p.mocap_pos, model.nmocap, 3)
        self._mocap_quat = _wrap_mjtNum_2d(p.mocap_quat, model.nmocap, 4)
        self._userdata = _wrap_mjtNum_1d(p.userdata, model.nuserdata)
        self._sensordata = _wrap_mjtNum_1d(p.sensordata, model.nsensordata)
        self._xpos = _wrap_mjtNum_2d(p.xpos, model.nbody, 3)
        self._xquat = _wrap_mjtNum_2d(p.xquat, model.nbody, 4)
        self._xmat = _wrap_mjtNum_2d(p.xmat, model.nbody, 9)
        self._xipos = _wrap_mjtNum_2d(p.xipos, model.nbody, 3)
        self._ximat = _wrap_mjtNum_2d(p.ximat, model.nbody, 9)
        self._xanchor = _wrap_mjtNum_2d(p.xanchor, model.njnt, 3)
        self._xaxis = _wrap_mjtNum_2d(p.xaxis, model.njnt, 3)
        self._geom_xpos = _wrap_mjtNum_2d(p.geom_xpos, model.ngeom, 3)
        self._geom_xmat = _wrap_mjtNum_2d(p.geom_xmat, model.ngeom, 9)
        self._site_xpos = _wrap_mjtNum_2d(p.site_xpos, model.nsite, 3)
        self._site_xmat = _wrap_mjtNum_2d(p.site_xmat, model.nsite, 9)
        self._cam_xpos = _wrap_mjtNum_2d(p.cam_xpos, model.ncam, 3)
        self._cam_xmat = _wrap_mjtNum_2d(p.cam_xmat, model.ncam, 9)
        self._light_xpos = _wrap_mjtNum_2d(p.light_xpos, model.nlight, 3)
        self._light_xdir = _wrap_mjtNum_2d(p.light_xdir, model.nlight, 3)
        self._subtree_com = _wrap_mjtNum_2d(p.subtree_com, model.nbody, 3)
        self._cdof = _wrap_mjtNum_2d(p.cdof, model.nv, 6)
        self._cinert = _wrap_mjtNum_2d(p.cinert, model.nbody, 10)
        self._ten_wrapadr = _wrap_int_1d(p.ten_wrapadr, model.ntendon)
        self._ten_wrapnum = _wrap_int_1d(p.ten_wrapnum, model.ntendon)
        self._ten_length = _wrap_mjtNum_1d(p.ten_length, model.ntendon)
        self._ten_moment = _wrap_mjtNum_2d(p.ten_moment, model.ntendon, model.nv)
        self._wrap_obj = _wrap_int_1d(p.wrap_obj, model.nwrap*2)
        self._wrap_xpos = _wrap_mjtNum_2d(p.wrap_xpos, model.nwrap*2, 3)
        self._actuator_length = _wrap_mjtNum_1d(p.actuator_length, model.nu)
        self._actuator_moment = _wrap_mjtNum_2d(p.actuator_moment, model.nu, model.nv)
        self._crb = _wrap_mjtNum_2d(p.crb, model.nbody, 10)
        self._qM = _wrap_mjtNum_1d(p.qM, model.nM)
        self._qLD = _wrap_mjtNum_1d(p.qLD, model.nM)
        self._qLDiagInv = _wrap_mjtNum_1d(p.qLDiagInv, model.nv)
        self._qLDiagSqrtInv = _wrap_mjtNum_1d(p.qLDiagSqrtInv, model.nv)
        self._contact = tuple([WrapMjContact(&p.contact[i]) for i in range(model.nconmax)])
        self._efc_type = _wrap_int_1d(p.efc_type, model.njmax)
        self._efc_id = _wrap_int_1d(p.efc_id, model.njmax)
        self._efc_rownnz = _wrap_int_1d(p.efc_rownnz, model.njmax)
        self._efc_rowadr = _wrap_int_1d(p.efc_rowadr, model.njmax)
        self._efc_colind = _wrap_int_2d(p.efc_colind, model.njmax, model.nv)
        self._efc_rownnz_T = _wrap_int_1d(p.efc_rownnz_T, model.nv)
        self._efc_rowadr_T = _wrap_int_1d(p.efc_rowadr_T, model.nv)
        self._efc_colind_T = _wrap_int_2d(p.efc_colind_T, model.nv, model.njmax)
        self._efc_solref = _wrap_mjtNum_2d(p.efc_solref, model.njmax, mjNREF)
        self._efc_solimp = _wrap_mjtNum_2d(p.efc_solimp, model.njmax, mjNIMP)
        self._efc_margin = _wrap_mjtNum_1d(p.efc_margin, model.njmax)
        self._efc_frictionloss = _wrap_mjtNum_1d(p.efc_frictionloss, model.njmax)
        self._efc_pos = _wrap_mjtNum_1d(p.efc_pos, model.njmax)
        self._efc_J = _wrap_mjtNum_2d(p.efc_J, model.njmax, model.nv)
        self._efc_J_T = _wrap_mjtNum_2d(p.efc_J_T, model.nv, model.njmax)
        self._efc_diagApprox = _wrap_mjtNum_1d(p.efc_diagApprox, model.njmax)
        self._efc_D = _wrap_mjtNum_1d(p.efc_D, model.njmax)
        self._efc_R = _wrap_mjtNum_1d(p.efc_R, model.njmax)
        self._efc_AR = _wrap_mjtNum_2d(p.efc_AR, model.njmax, model.njmax)
        self._e_ARchol = _wrap_mjtNum_2d(p.e_ARchol, model.nemax, model.nemax)
        self._fc_e_rect = _wrap_mjtNum_2d(p.fc_e_rect, model.njmax, model.nemax)
        self._fc_AR = _wrap_mjtNum_2d(p.fc_AR, model.njmax, model.njmax)
        self._ten_velocity = _wrap_mjtNum_1d(p.ten_velocity, model.ntendon)
        self._actuator_velocity = _wrap_mjtNum_1d(p.actuator_velocity, model.nu)
        self._cvel = _wrap_mjtNum_2d(p.cvel, model.nbody, 6)
        self._cdof_dot = _wrap_mjtNum_2d(p.cdof_dot, model.nv, 6)
        self._qfrc_bias = _wrap_mjtNum_1d(p.qfrc_bias, model.nv)
        self._qfrc_passive = _wrap_mjtNum_1d(p.qfrc_passive, model.nv)
        self._efc_vel = _wrap_mjtNum_1d(p.efc_vel, model.njmax)
        self._efc_aref = _wrap_mjtNum_1d(p.efc_aref, model.njmax)
        self._subtree_linvel = _wrap_mjtNum_2d(p.subtree_linvel, model.nbody, 3)
        self._subtree_angmom = _wrap_mjtNum_2d(p.subtree_angmom, model.nbody, 3)
        self._actuator_force = _wrap_mjtNum_1d(p.actuator_force, model.nu)
        self._qfrc_actuator = _wrap_mjtNum_1d(p.qfrc_actuator, model.nv)
        self._qfrc_unc = _wrap_mjtNum_1d(p.qfrc_unc, model.nv)
        self._qacc_unc = _wrap_mjtNum_1d(p.qacc_unc, model.nv)
        self._efc_b = _wrap_mjtNum_1d(p.efc_b, model.njmax)
        self._fc_b = _wrap_mjtNum_1d(p.fc_b, model.njmax)
        self._efc_force = _wrap_mjtNum_1d(p.efc_force, model.njmax)
        self._qfrc_constraint = _wrap_mjtNum_1d(p.qfrc_constraint, model.nv)
        self._qfrc_inverse = _wrap_mjtNum_1d(p.qfrc_inverse, model.nv)
        self._cacc = _wrap_mjtNum_2d(p.cacc, model.nbody, 6)
        self._cfrc_int = _wrap_mjtNum_2d(p.cfrc_int, model.nbody, 6)
        self._cfrc_ext = _wrap_mjtNum_2d(p.cfrc_ext, model.nbody, 6)
        self._nwarning = _wrap_int_1d(&p.nwarning[0], mjNWARNING)
        self._warning_info = _wrap_int_1d(&p.warning_info[0], mjNWARNING)
        self._timer_ncall = _wrap_int_1d(&p.timer_ncall[0], mjNTIMER)
        self._timer_duration = _wrap_mjtNum_1d(&p.timer_duration[0], mjNTIMER)
        self._solver_trace = _wrap_mjtNum_1d(&p.solver_trace[0], 200)
        self._solver_fwdinv = _wrap_mjtNum_1d(&p.solver_fwdinv[0], 2)
        self._energy = _wrap_mjtNum_1d(&p.energy[0], 2)
    @property
    def nstack(self): return self.ptr.nstack
    @nstack.setter
    def nstack(self, int x): self.ptr.nstack = x
    @property
    def nbuffer(self): return self.ptr.nbuffer
    @nbuffer.setter
    def nbuffer(self, int x): self.ptr.nbuffer = x
    @property
    def pstack(self): return self.ptr.pstack
    @pstack.setter
    def pstack(self, int x): self.ptr.pstack = x
    @property
    def maxuse_stack(self): return self.ptr.maxuse_stack
    @maxuse_stack.setter
    def maxuse_stack(self, int x): self.ptr.maxuse_stack = x
    @property
    def maxuse_con(self): return self.ptr.maxuse_con
    @maxuse_con.setter
    def maxuse_con(self, int x): self.ptr.maxuse_con = x
    @property
    def maxuse_efc(self): return self.ptr.maxuse_efc
    @maxuse_efc.setter
    def maxuse_efc(self, int x): self.ptr.maxuse_efc = x
    @property
    def solver_iter(self): return self.ptr.solver_iter
    @solver_iter.setter
    def solver_iter(self, int x): self.ptr.solver_iter = x
    @property
    def ne(self): return self.ptr.ne
    @ne.setter
    def ne(self, int x): self.ptr.ne = x
    @property
    def nf(self): return self.ptr.nf
    @nf.setter
    def nf(self, int x): self.ptr.nf = x
    @property
    def nefc(self): return self.ptr.nefc
    @nefc.setter
    def nefc(self, int x): self.ptr.nefc = x
    @property
    def ncon(self): return self.ptr.ncon
    @ncon.setter
    def ncon(self, int x): self.ptr.ncon = x
    @property
    def time(self): return self.ptr.time
    @time.setter
    def time(self, mjtNum x): self.ptr.time = x
    @property
    def qpos(self): return self._qpos
    @property
    def qvel(self): return self._qvel
    @property
    def act(self): return self._act
    @property
    def ctrl(self): return self._ctrl
    @property
    def qfrc_applied(self): return self._qfrc_applied
    @property
    def xfrc_applied(self): return self._xfrc_applied
    @property
    def qacc(self): return self._qacc
    @property
    def act_dot(self): return self._act_dot
    @property
    def mocap_pos(self): return self._mocap_pos
    @property
    def mocap_quat(self): return self._mocap_quat
    @property
    def userdata(self): return self._userdata
    @property
    def sensordata(self): return self._sensordata
    @property
    def xpos(self): return self._xpos
    @property
    def xquat(self): return self._xquat
    @property
    def xmat(self): return self._xmat
    @property
    def xipos(self): return self._xipos
    @property
    def ximat(self): return self._ximat
    @property
    def xanchor(self): return self._xanchor
    @property
    def xaxis(self): return self._xaxis
    @property
    def geom_xpos(self): return self._geom_xpos
    @property
    def geom_xmat(self): return self._geom_xmat
    @property
    def site_xpos(self): return self._site_xpos
    @property
    def site_xmat(self): return self._site_xmat
    @property
    def cam_xpos(self): return self._cam_xpos
    @property
    def cam_xmat(self): return self._cam_xmat
    @property
    def light_xpos(self): return self._light_xpos
    @property
    def light_xdir(self): return self._light_xdir
    @property
    def subtree_com(self): return self._subtree_com
    @property
    def cdof(self): return self._cdof
    @property
    def cinert(self): return self._cinert
    @property
    def ten_wrapadr(self): return self._ten_wrapadr
    @property
    def ten_wrapnum(self): return self._ten_wrapnum
    @property
    def ten_length(self): return self._ten_length
    @property
    def ten_moment(self): return self._ten_moment
    @property
    def wrap_obj(self): return self._wrap_obj
    @property
    def wrap_xpos(self): return self._wrap_xpos
    @property
    def actuator_length(self): return self._actuator_length
    @property
    def actuator_moment(self): return self._actuator_moment
    @property
    def crb(self): return self._crb
    @property
    def qM(self): return self._qM
    @property
    def qLD(self): return self._qLD
    @property
    def qLDiagInv(self): return self._qLDiagInv
    @property
    def qLDiagSqrtInv(self): return self._qLDiagSqrtInv
    @property
    def contact(self): return self._contact
    @property
    def efc_type(self): return self._efc_type
    @property
    def efc_id(self): return self._efc_id
    @property
    def efc_rownnz(self): return self._efc_rownnz
    @property
    def efc_rowadr(self): return self._efc_rowadr
    @property
    def efc_colind(self): return self._efc_colind
    @property
    def efc_rownnz_T(self): return self._efc_rownnz_T
    @property
    def efc_rowadr_T(self): return self._efc_rowadr_T
    @property
    def efc_colind_T(self): return self._efc_colind_T
    @property
    def efc_solref(self): return self._efc_solref
    @property
    def efc_solimp(self): return self._efc_solimp
    @property
    def efc_margin(self): return self._efc_margin
    @property
    def efc_frictionloss(self): return self._efc_frictionloss
    @property
    def efc_pos(self): return self._efc_pos
    @property
    def efc_J(self): return self._efc_J
    @property
    def efc_J_T(self): return self._efc_J_T
    @property
    def efc_diagApprox(self): return self._efc_diagApprox
    @property
    def efc_D(self): return self._efc_D
    @property
    def efc_R(self): return self._efc_R
    @property
    def efc_AR(self): return self._efc_AR
    @property
    def e_ARchol(self): return self._e_ARchol
    @property
    def fc_e_rect(self): return self._fc_e_rect
    @property
    def fc_AR(self): return self._fc_AR
    @property
    def ten_velocity(self): return self._ten_velocity
    @property
    def actuator_velocity(self): return self._actuator_velocity
    @property
    def cvel(self): return self._cvel
    @property
    def cdof_dot(self): return self._cdof_dot
    @property
    def qfrc_bias(self): return self._qfrc_bias
    @property
    def qfrc_passive(self): return self._qfrc_passive
    @property
    def efc_vel(self): return self._efc_vel
    @property
    def efc_aref(self): return self._efc_aref
    @property
    def subtree_linvel(self): return self._subtree_linvel
    @property
    def subtree_angmom(self): return self._subtree_angmom
    @property
    def actuator_force(self): return self._actuator_force
    @property
    def qfrc_actuator(self): return self._qfrc_actuator
    @property
    def qfrc_unc(self): return self._qfrc_unc
    @property
    def qacc_unc(self): return self._qacc_unc
    @property
    def efc_b(self): return self._efc_b
    @property
    def fc_b(self): return self._fc_b
    @property
    def efc_force(self): return self._efc_force
    @property
    def qfrc_constraint(self): return self._qfrc_constraint
    @property
    def qfrc_inverse(self): return self._qfrc_inverse
    @property
    def cacc(self): return self._cacc
    @property
    def cfrc_int(self): return self._cfrc_int
    @property
    def cfrc_ext(self): return self._cfrc_ext
    @property
    def nwarning(self): return self._nwarning
    @property
    def warning_info(self): return self._warning_info
    @property
    def timer_ncall(self): return self._timer_ncall
    @property
    def timer_duration(self): return self._timer_duration
    @property
    def solver_trace(self): return self._solver_trace
    @property
    def solver_fwdinv(self): return self._solver_fwdinv
    @property
    def energy(self): return self._energy

cdef PyMjData WrapMjData(mjData* p, mjModel* model):
    cdef PyMjData o = PyMjData()
    o._set(p, model)
    return o

cdef class PyMjvPerturb(object):
    cdef mjvPerturb* ptr
    cdef np.ndarray _refpos
    cdef np.ndarray _refquat
    cdef np.ndarray _localpos
    cdef void _set(self, mjvPerturb* p):
        self.ptr = p
        self._refpos = _wrap_mjtNum_1d(&p.refpos[0], 3)
        self._refquat = _wrap_mjtNum_1d(&p.refquat[0], 4)
        self._localpos = _wrap_mjtNum_1d(&p.localpos[0], 3)
    @property
    def select(self): return self.ptr.select
    @select.setter
    def select(self, int x): self.ptr.select = x
    @property
    def active(self): return self.ptr.active
    @active.setter
    def active(self, int x): self.ptr.active = x
    @property
    def scale(self): return self.ptr.scale
    @scale.setter
    def scale(self, mjtNum x): self.ptr.scale = x
    @property
    def refpos(self): return self._refpos
    @property
    def refquat(self): return self._refquat
    @property
    def localpos(self): return self._localpos

cdef PyMjvPerturb WrapMjvPerturb(mjvPerturb* p):
    cdef PyMjvPerturb o = PyMjvPerturb()
    o._set(p)
    return o

cdef class PyMjvCamera(object):
    cdef mjvCamera* ptr
    cdef np.ndarray _lookat
    cdef void _set(self, mjvCamera* p):
        self.ptr = p
        self._lookat = _wrap_mjtNum_1d(&p.lookat[0], 3)
    @property
    def type(self): return self.ptr.type
    @type.setter
    def type(self, int x): self.ptr.type = x
    @property
    def fixedcamid(self): return self.ptr.fixedcamid
    @fixedcamid.setter
    def fixedcamid(self, int x): self.ptr.fixedcamid = x
    @property
    def trackbodyid(self): return self.ptr.trackbodyid
    @trackbodyid.setter
    def trackbodyid(self, int x): self.ptr.trackbodyid = x
    @property
    def distance(self): return self.ptr.distance
    @distance.setter
    def distance(self, mjtNum x): self.ptr.distance = x
    @property
    def azimuth(self): return self.ptr.azimuth
    @azimuth.setter
    def azimuth(self, mjtNum x): self.ptr.azimuth = x
    @property
    def elevation(self): return self.ptr.elevation
    @elevation.setter
    def elevation(self, mjtNum x): self.ptr.elevation = x
    @property
    def lookat(self): return self._lookat

cdef PyMjvCamera WrapMjvCamera(mjvCamera* p):
    cdef PyMjvCamera o = PyMjvCamera()
    o._set(p)
    return o

cdef class PyMjvGLCamera(object):
    cdef mjvGLCamera* ptr
    cdef np.ndarray _pos
    cdef np.ndarray _forward
    cdef np.ndarray _up
    cdef void _set(self, mjvGLCamera* p):
        self.ptr = p
        self._pos = _wrap_float_1d(&p.pos[0], 3)
        self._forward = _wrap_float_1d(&p.forward[0], 3)
        self._up = _wrap_float_1d(&p.up[0], 3)
    @property
    def frustum_center(self): return self.ptr.frustum_center
    @frustum_center.setter
    def frustum_center(self, float x): self.ptr.frustum_center = x
    @property
    def frustum_bottom(self): return self.ptr.frustum_bottom
    @frustum_bottom.setter
    def frustum_bottom(self, float x): self.ptr.frustum_bottom = x
    @property
    def frustum_top(self): return self.ptr.frustum_top
    @frustum_top.setter
    def frustum_top(self, float x): self.ptr.frustum_top = x
    @property
    def frustum_near(self): return self.ptr.frustum_near
    @frustum_near.setter
    def frustum_near(self, float x): self.ptr.frustum_near = x
    @property
    def frustum_far(self): return self.ptr.frustum_far
    @frustum_far.setter
    def frustum_far(self, float x): self.ptr.frustum_far = x
    @property
    def pos(self): return self._pos
    @property
    def forward(self): return self._forward
    @property
    def up(self): return self._up

cdef PyMjvGLCamera WrapMjvGLCamera(mjvGLCamera* p):
    cdef PyMjvGLCamera o = PyMjvGLCamera()
    o._set(p)
    return o

cdef class PyMjvGeom(object):
    cdef mjvGeom* ptr
    cdef np.ndarray _texrepeat
    cdef np.ndarray _size
    cdef np.ndarray _pos
    cdef np.ndarray _mat
    cdef np.ndarray _rgba
    cdef np.ndarray _label
    cdef void _set(self, mjvGeom* p):
        self.ptr = p
        self._texrepeat = _wrap_float_1d(&p.texrepeat[0], 2)
        self._size = _wrap_float_1d(&p.size[0], 3)
        self._pos = _wrap_float_1d(&p.pos[0], 3)
        self._mat = _wrap_float_1d(&p.mat[0], 9)
        self._rgba = _wrap_float_1d(&p.rgba[0], 4)
        self._label = _wrap_char_1d(&p.label[0], 100)
    @property
    def type(self): return self.ptr.type
    @type.setter
    def type(self, int x): self.ptr.type = x
    @property
    def dataid(self): return self.ptr.dataid
    @dataid.setter
    def dataid(self, int x): self.ptr.dataid = x
    @property
    def objtype(self): return self.ptr.objtype
    @objtype.setter
    def objtype(self, int x): self.ptr.objtype = x
    @property
    def objid(self): return self.ptr.objid
    @objid.setter
    def objid(self, int x): self.ptr.objid = x
    @property
    def category(self): return self.ptr.category
    @category.setter
    def category(self, int x): self.ptr.category = x
    @property
    def texid(self): return self.ptr.texid
    @texid.setter
    def texid(self, int x): self.ptr.texid = x
    @property
    def texuniform(self): return self.ptr.texuniform
    @texuniform.setter
    def texuniform(self, int x): self.ptr.texuniform = x
    @property
    def emission(self): return self.ptr.emission
    @emission.setter
    def emission(self, float x): self.ptr.emission = x
    @property
    def specular(self): return self.ptr.specular
    @specular.setter
    def specular(self, float x): self.ptr.specular = x
    @property
    def shininess(self): return self.ptr.shininess
    @shininess.setter
    def shininess(self, float x): self.ptr.shininess = x
    @property
    def reflectance(self): return self.ptr.reflectance
    @reflectance.setter
    def reflectance(self, float x): self.ptr.reflectance = x
    @property
    def camdist(self): return self.ptr.camdist
    @camdist.setter
    def camdist(self, float x): self.ptr.camdist = x
    @property
    def rbound(self): return self.ptr.rbound
    @rbound.setter
    def rbound(self, float x): self.ptr.rbound = x
    @property
    def transparent(self): return self.ptr.transparent
    @transparent.setter
    def transparent(self, mjtByte x): self.ptr.transparent = x
    @property
    def texrepeat(self): return self._texrepeat
    @property
    def size(self): return self._size
    @property
    def pos(self): return self._pos
    @property
    def mat(self): return self._mat
    @property
    def rgba(self): return self._rgba
    @property
    def label(self): return self._label

cdef PyMjvGeom WrapMjvGeom(mjvGeom* p):
    cdef PyMjvGeom o = PyMjvGeom()
    o._set(p)
    return o

cdef class PyMjvLight(object):
    cdef mjvLight* ptr
    cdef np.ndarray _pos
    cdef np.ndarray _dir
    cdef np.ndarray _attenuation
    cdef np.ndarray _ambient
    cdef np.ndarray _diffuse
    cdef np.ndarray _specular
    cdef void _set(self, mjvLight* p):
        self.ptr = p
        self._pos = _wrap_float_1d(&p.pos[0], 3)
        self._dir = _wrap_float_1d(&p.dir[0], 3)
        self._attenuation = _wrap_float_1d(&p.attenuation[0], 3)
        self._ambient = _wrap_float_1d(&p.ambient[0], 3)
        self._diffuse = _wrap_float_1d(&p.diffuse[0], 3)
        self._specular = _wrap_float_1d(&p.specular[0], 3)
    @property
    def cutoff(self): return self.ptr.cutoff
    @cutoff.setter
    def cutoff(self, float x): self.ptr.cutoff = x
    @property
    def exponent(self): return self.ptr.exponent
    @exponent.setter
    def exponent(self, float x): self.ptr.exponent = x
    @property
    def headlight(self): return self.ptr.headlight
    @headlight.setter
    def headlight(self, mjtByte x): self.ptr.headlight = x
    @property
    def directional(self): return self.ptr.directional
    @directional.setter
    def directional(self, mjtByte x): self.ptr.directional = x
    @property
    def castshadow(self): return self.ptr.castshadow
    @castshadow.setter
    def castshadow(self, mjtByte x): self.ptr.castshadow = x
    @property
    def pos(self): return self._pos
    @property
    def dir(self): return self._dir
    @property
    def attenuation(self): return self._attenuation
    @property
    def ambient(self): return self._ambient
    @property
    def diffuse(self): return self._diffuse
    @property
    def specular(self): return self._specular

cdef PyMjvLight WrapMjvLight(mjvLight* p):
    cdef PyMjvLight o = PyMjvLight()
    o._set(p)
    return o

cdef class PyMjvOption(object):
    cdef mjvOption* ptr
    cdef np.ndarray _geomgroup
    cdef np.ndarray _sitegroup
    cdef np.ndarray _flags
    cdef void _set(self, mjvOption* p):
        self.ptr = p
        self._geomgroup = _wrap_mjtByte_1d(&p.geomgroup[0], 5)
        self._sitegroup = _wrap_mjtByte_1d(&p.sitegroup[0], 5)
        self._flags = _wrap_mjtByte_1d(&p.flags[0], mjNVISFLAG)
    @property
    def label(self): return self.ptr.label
    @label.setter
    def label(self, int x): self.ptr.label = x
    @property
    def frame(self): return self.ptr.frame
    @frame.setter
    def frame(self, int x): self.ptr.frame = x
    @property
    def geomgroup(self): return self._geomgroup
    @property
    def sitegroup(self): return self._sitegroup
    @property
    def flags(self): return self._flags

cdef PyMjvOption WrapMjvOption(mjvOption* p):
    cdef PyMjvOption o = PyMjvOption()
    o._set(p)
    return o

cdef class PyMjvScene(object):
    cdef mjvScene* ptr
    cdef np.ndarray _translate
    cdef np.ndarray _rotate
    cdef np.ndarray _flags
    cdef void _set(self, mjvScene* p):
        self.ptr = p
        self._translate = _wrap_float_1d(&p.translate[0], 3)
        self._rotate = _wrap_float_1d(&p.rotate[0], 4)
        self._flags = _wrap_mjtByte_1d(&p.flags[0], mjNRNDFLAG)
    @property
    def maxgeom(self): return self.ptr.maxgeom
    @maxgeom.setter
    def maxgeom(self, int x): self.ptr.maxgeom = x
    @property
    def ngeom(self): return self.ptr.ngeom
    @ngeom.setter
    def ngeom(self, int x): self.ptr.ngeom = x
    @property
    def nlight(self): return self.ptr.nlight
    @nlight.setter
    def nlight(self, int x): self.ptr.nlight = x
    @property
    def enabletransform(self): return self.ptr.enabletransform
    @enabletransform.setter
    def enabletransform(self, mjtByte x): self.ptr.enabletransform = x
    @property
    def scale(self): return self.ptr.scale
    @scale.setter
    def scale(self, float x): self.ptr.scale = x
    @property
    def stereo(self): return self.ptr.stereo
    @stereo.setter
    def stereo(self, int x): self.ptr.stereo = x
    @property
    def translate(self): return self._translate
    @property
    def rotate(self): return self._rotate
    @property
    def flags(self): return self._flags

cdef PyMjvScene WrapMjvScene(mjvScene* p):
    cdef PyMjvScene o = PyMjvScene()
    o._set(p)
    return o

cdef class PyMjrRect(object):
    cdef mjrRect* ptr
    cdef void _set(self, mjrRect* p):
        self.ptr = p
    @property
    def left(self): return self.ptr.left
    @left.setter
    def left(self, int x): self.ptr.left = x
    @property
    def bottom(self): return self.ptr.bottom
    @bottom.setter
    def bottom(self, int x): self.ptr.bottom = x
    @property
    def width(self): return self.ptr.width
    @width.setter
    def width(self, int x): self.ptr.width = x
    @property
    def height(self): return self.ptr.height
    @height.setter
    def height(self, int x): self.ptr.height = x

cdef PyMjrRect WrapMjrRect(mjrRect* p):
    cdef PyMjrRect o = PyMjrRect()
    o._set(p)
    return o

cdef class PyMjrContext(object):
    cdef mjrContext* ptr
    cdef np.ndarray _textureType
    cdef np.ndarray _texture
    cdef np.ndarray _charWidth
    cdef np.ndarray _charWidthBig
    cdef void _set(self, mjrContext* p):
        self.ptr = p
        self._textureType = _wrap_int_1d(&p.textureType[0], 100)
        self._texture = _wrap_unsigned_int_1d(&p.texture[0], 100)
        self._charWidth = _wrap_int_1d(&p.charWidth[0], 127)
        self._charWidthBig = _wrap_int_1d(&p.charWidthBig[0], 127)
    @property
    def lineWidth(self): return self.ptr.lineWidth
    @lineWidth.setter
    def lineWidth(self, float x): self.ptr.lineWidth = x
    @property
    def shadowClip(self): return self.ptr.shadowClip
    @shadowClip.setter
    def shadowClip(self, float x): self.ptr.shadowClip = x
    @property
    def shadowScale(self): return self.ptr.shadowScale
    @shadowScale.setter
    def shadowScale(self, float x): self.ptr.shadowScale = x
    @property
    def shadowSize(self): return self.ptr.shadowSize
    @shadowSize.setter
    def shadowSize(self, int x): self.ptr.shadowSize = x
    @property
    def offWidth(self): return self.ptr.offWidth
    @offWidth.setter
    def offWidth(self, int x): self.ptr.offWidth = x
    @property
    def offHeight(self): return self.ptr.offHeight
    @offHeight.setter
    def offHeight(self, int x): self.ptr.offHeight = x
    @property
    def offSamples(self): return self.ptr.offSamples
    @offSamples.setter
    def offSamples(self, int x): self.ptr.offSamples = x
    @property
    def offFBO(self): return self.ptr.offFBO
    @offFBO.setter
    def offFBO(self, unsigned int x): self.ptr.offFBO = x
    @property
    def offFBO_r(self): return self.ptr.offFBO_r
    @offFBO_r.setter
    def offFBO_r(self, unsigned int x): self.ptr.offFBO_r = x
    @property
    def offColor(self): return self.ptr.offColor
    @offColor.setter
    def offColor(self, unsigned int x): self.ptr.offColor = x
    @property
    def offColor_r(self): return self.ptr.offColor_r
    @offColor_r.setter
    def offColor_r(self, unsigned int x): self.ptr.offColor_r = x
    @property
    def offDepthStencil(self): return self.ptr.offDepthStencil
    @offDepthStencil.setter
    def offDepthStencil(self, unsigned int x): self.ptr.offDepthStencil = x
    @property
    def offDepthStencil_r(self): return self.ptr.offDepthStencil_r
    @offDepthStencil_r.setter
    def offDepthStencil_r(self, unsigned int x): self.ptr.offDepthStencil_r = x
    @property
    def shadowFBO(self): return self.ptr.shadowFBO
    @shadowFBO.setter
    def shadowFBO(self, unsigned int x): self.ptr.shadowFBO = x
    @property
    def shadowTex(self): return self.ptr.shadowTex
    @shadowTex.setter
    def shadowTex(self, unsigned int x): self.ptr.shadowTex = x
    @property
    def ntexture(self): return self.ptr.ntexture
    @ntexture.setter
    def ntexture(self, int x): self.ptr.ntexture = x
    @property
    def basePlane(self): return self.ptr.basePlane
    @basePlane.setter
    def basePlane(self, unsigned int x): self.ptr.basePlane = x
    @property
    def baseMesh(self): return self.ptr.baseMesh
    @baseMesh.setter
    def baseMesh(self, unsigned int x): self.ptr.baseMesh = x
    @property
    def baseHField(self): return self.ptr.baseHField
    @baseHField.setter
    def baseHField(self, unsigned int x): self.ptr.baseHField = x
    @property
    def baseBuiltin(self): return self.ptr.baseBuiltin
    @baseBuiltin.setter
    def baseBuiltin(self, unsigned int x): self.ptr.baseBuiltin = x
    @property
    def baseFontNormal(self): return self.ptr.baseFontNormal
    @baseFontNormal.setter
    def baseFontNormal(self, unsigned int x): self.ptr.baseFontNormal = x
    @property
    def baseFontShadow(self): return self.ptr.baseFontShadow
    @baseFontShadow.setter
    def baseFontShadow(self, unsigned int x): self.ptr.baseFontShadow = x
    @property
    def baseFontBig(self): return self.ptr.baseFontBig
    @baseFontBig.setter
    def baseFontBig(self, unsigned int x): self.ptr.baseFontBig = x
    @property
    def rangePlane(self): return self.ptr.rangePlane
    @rangePlane.setter
    def rangePlane(self, int x): self.ptr.rangePlane = x
    @property
    def rangeMesh(self): return self.ptr.rangeMesh
    @rangeMesh.setter
    def rangeMesh(self, int x): self.ptr.rangeMesh = x
    @property
    def rangeHField(self): return self.ptr.rangeHField
    @rangeHField.setter
    def rangeHField(self, int x): self.ptr.rangeHField = x
    @property
    def rangeBuiltin(self): return self.ptr.rangeBuiltin
    @rangeBuiltin.setter
    def rangeBuiltin(self, int x): self.ptr.rangeBuiltin = x
    @property
    def rangeFont(self): return self.ptr.rangeFont
    @rangeFont.setter
    def rangeFont(self, int x): self.ptr.rangeFont = x
    @property
    def charHeight(self): return self.ptr.charHeight
    @charHeight.setter
    def charHeight(self, int x): self.ptr.charHeight = x
    @property
    def charHeightBig(self): return self.ptr.charHeightBig
    @charHeightBig.setter
    def charHeightBig(self, int x): self.ptr.charHeightBig = x
    @property
    def glewInitialized(self): return self.ptr.glewInitialized
    @glewInitialized.setter
    def glewInitialized(self, int x): self.ptr.glewInitialized = x
    @property
    def windowAvailable(self): return self.ptr.windowAvailable
    @windowAvailable.setter
    def windowAvailable(self, int x): self.ptr.windowAvailable = x
    @property
    def windowSamples(self): return self.ptr.windowSamples
    @windowSamples.setter
    def windowSamples(self, int x): self.ptr.windowSamples = x
    @property
    def windowStereo(self): return self.ptr.windowStereo
    @windowStereo.setter
    def windowStereo(self, int x): self.ptr.windowStereo = x
    @property
    def windowDoublebuffer(self): return self.ptr.windowDoublebuffer
    @windowDoublebuffer.setter
    def windowDoublebuffer(self, int x): self.ptr.windowDoublebuffer = x
    @property
    def currentBuffer(self): return self.ptr.currentBuffer
    @currentBuffer.setter
    def currentBuffer(self, int x): self.ptr.currentBuffer = x
    @property
    def textureType(self): return self._textureType
    @property
    def texture(self): return self._texture
    @property
    def charWidth(self): return self._charWidth
    @property
    def charWidthBig(self): return self._charWidthBig

cdef PyMjrContext WrapMjrContext(mjrContext* p):
    cdef PyMjrContext o = PyMjrContext()
    o._set(p)
    return o

cdef inline np.ndarray _wrap_char_1d(char* a, int shape0):
    if shape0 == 0: return None
    cdef char[:] b = <char[:shape0]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_float_1d(float* a, int shape0):
    if shape0 == 0: return None
    cdef float[:] b = <float[:shape0]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_int_1d(int* a, int shape0):
    if shape0 == 0: return None
    cdef int[:] b = <int[:shape0]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_mjtByte_1d(mjtByte* a, int shape0):
    if shape0 == 0: return None
    cdef mjtByte[:] b = <mjtByte[:shape0]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_mjtNum_1d(mjtNum* a, int shape0):
    if shape0 == 0: return None
    cdef mjtNum[:] b = <mjtNum[:shape0]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_unsigned_int_1d(unsigned int* a, int shape0):
    if shape0 == 0: return None
    cdef unsigned int[:] b = <unsigned int[:shape0]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_char_2d(char* a, int shape0, int shape1):
    if shape0 * shape1 == 0: return None
    cdef char[:,:] b = <char[:shape0,:shape1]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_float_2d(float* a, int shape0, int shape1):
    if shape0 * shape1 == 0: return None
    cdef float[:,:] b = <float[:shape0,:shape1]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_int_2d(int* a, int shape0, int shape1):
    if shape0 * shape1 == 0: return None
    cdef int[:,:] b = <int[:shape0,:shape1]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_mjtByte_2d(mjtByte* a, int shape0, int shape1):
    if shape0 * shape1 == 0: return None
    cdef mjtByte[:,:] b = <mjtByte[:shape0,:shape1]> a
    return np.asarray(b)

cdef inline np.ndarray _wrap_mjtNum_2d(mjtNum* a, int shape0, int shape1):
    if shape0 * shape1 == 0: return None
    cdef mjtNum[:,:] b = <mjtNum[:shape0,:shape1]> a
    return np.asarray(b)
