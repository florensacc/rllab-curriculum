# cython: language_level=3
import ctypes

import cython
import math

include "wrappers.pxi"

import logging

from libc.string cimport strncpy
cimport cython.parallel
cimport openmp
from cython.parallel import parallel, prange, threadid
import numpy as np
cimport numpy as np

from libc cimport math as cmath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Constants(object):
    GRID_TOPLEFT = mjGRID_TOPLEFT
    GRID_TOPRIGHT = mjGRID_TOPRIGHT
    GRID_BOTTOMLEFT = mjGRID_BOTTOMLEFT
    GRID_BOTTOMRIGHT = mjGRID_BOTTOMRIGHT

    FONT_NORMAL = mjFONT_NORMAL
    FONT_SHADOW = mjFONT_SHADOW
    FONT_BIG = mjFONT_BIG

    MOUSE_NONE = mjMOUSE_NONE
    MOUSE_ROTATE_V = mjMOUSE_ROTATE_V
    MOUSE_ROTATE_H = mjMOUSE_ROTATE_H
    MOUSE_MOVE_V = mjMOUSE_MOVE_V
    MOUSE_MOVE_H = mjMOUSE_MOVE_H
    MOUSE_ZOOM = mjMOUSE_ZOOM
    MOUSE_SELECT = mjMOUSE_SELECT

    CAMERA_FREE = mjCAMERA_FREE
    CAMERA_TRACKING = mjCAMERA_TRACKING
    CAMERA_FIXED = mjCAMERA_FIXED
    CAMERA_USER = mjCAMERA_USER

    GEOM_PLANE = mjGEOM_PLANE
    GEOM_HFIELD = mjGEOM_HFIELD
    GEOM_SPHERE = mjGEOM_SPHERE
    GEOM_CAPSULE = mjGEOM_CAPSULE
    GEOM_ELLIPSOID = mjGEOM_ELLIPSOID
    GEOM_CYLINDER = mjGEOM_CYLINDER
    GEOM_BOX = mjGEOM_BOX
    GEOM_MESH = mjGEOM_MESH
    NGEOMTYPES = mjNGEOMTYPES
    GEOM_ARROW = mjGEOM_ARROW
    GEOM_ARROW1 = mjGEOM_ARROW1
    GEOM_ARROW2 = mjGEOM_ARROW2
    GEOM_LABEL = mjGEOM_LABEL
    GEOM_NONE = mjGEOM_NONE

    NLABEL = mjNLABEL
    NFRAME = mjNFRAME

    @staticmethod
    def disable_string(int i):
        assert 0 <= i < mjNDISABLE
        return mjDISABLESTRING[i]
    @staticmethod
    def enable_string(int i):
        assert 0 <= i < mjNENABLE
        return mjENABLESTRING[i]
    @staticmethod
    def timer_string(int i):
        assert 0 <= i < mjNTIMER
        return mjTIMERSTRING[i]
    @staticmethod
    def label_string(int i):
        assert 0 <= i < mjNLABEL
        return mjLABELSTRING[i]
    @staticmethod
    def frame_string(int i):
        assert 0 <= i < mjNFRAME
        return mjFRAMESTRING[i]


def activate(license_path):
    return mj_activate(license_path.encode())

cdef inline tuple _extract_mj_names(char*names, int*name_adr, int n):
    return tuple([(names + name_adr[i]).decode() for i in range(n)])

cdef class MjSim(object):
    """
    Keeps track of mjModel and mjData and provides some helper utilities
    """

    cdef mjModel *_model
    cdef mjData *_data

    # Exposed in Python for the user's convenience
    cdef readonly PyMjModel model
    cdef readonly PyMjData data
    cdef readonly tuple body_names, joint_names, geom_names, site_names

    def __cinit__(self, str xml):
        logger.debug('mj_loadXML')
        cdef char errstr[300]
        self._model = mj_loadXML(NULL, xml.encode(), errstr, 300)
        if self._model == NULL:
            logger.error('Tried to load this XML: %s' % xml)
            raise Exception('mj_loadXML error: %s' % errstr)
        logger.debug('mj_loadXML completed')

        self._data = mj_makeData(self._model)
        if self._data == NULL:
            if self._model != NULL:
                mj_deleteModel(self._model)
            raise Exception('mj_makeData failed!')

        self.model = WrapMjModel(self._model)
        self.data = WrapMjData(self._data, self._model)
        self.body_names = _extract_mj_names(self._model.names, self._model.name_bodyadr, self._model.nbody)
        self.joint_names = _extract_mj_names(self._model.names, self._model.name_jntadr, self._model.njnt)
        self.geom_names = _extract_mj_names(self._model.names, self._model.name_geomadr, self._model.ngeom)
        self.site_names = _extract_mj_names(self._model.names, self._model.name_siteadr, self._model.nsite)

    def __dealloc__(self):
        if self._data != NULL:
            mj_deleteData(self._data)
        if self._model != NULL:
            mj_deleteModel(self._model)

    def reset_data(self):
        mj_resetData(self._model, self._data)

    def forward(self):
        mj_forward(self._model, self._data)

    def step(self, int n=1):
        for _ in range(n):
            mj_step(self._model, self._data)

    @property
    def convergence(self):
        return self._data.solver_trace[max(0, min(self._data.solver_iter - 1, mjNTRACE - 1))]

cdef class MjViewerContext(object):
    cdef mjModel*_m
    cdef mjData*_d
    cdef mjvScene _scn
    cdef mjvCamera _cam
    cdef mjvOption _vopt
    cdef mjvPerturb _pert
    cdef mjrContext _con
    cdef bint _closed

    # Public wrappers
    cdef readonly PyMjvScene scn
    cdef readonly PyMjvCamera cam
    cdef readonly PyMjvOption vopt
    cdef readonly PyMjvPerturb pert
    cdef readonly PyMjrContext con

    def __cinit__(self, MjSim sim, int maxgeom=1000):
        self._m = sim.model.ptr
        self._d = sim.data.ptr
        self._pert.active = 0
        self._pert.select = 0
        mjv_makeScene(&self._scn, maxgeom)
        mjv_defaultCamera(&self._cam)
        mjv_defaultOption(&self._vopt)
        mjr_defaultContext(&self._con)
        mjr_makeContext(self._m, &self._con, mjFONTSCALE_150)
        self._closed = False
        self.scn = WrapMjvScene(&self._scn)
        self.cam = WrapMjvCamera(&self._cam)
        self.vopt = WrapMjvOption(&self._vopt)
        self.pert = WrapMjvPerturb(&self._pert)
        self.con = WrapMjrContext(&self._con)

    def __dealloc__(self):
        self.close()

    cpdef close(self):
        if self._closed: return
        mjr_freeContext(&self._con)
        mjv_freeScene(&self._scn)
        self._closed = True

    def update_scene(self):
        mjv_updateScene(self._m, self._d, &self._vopt, &self._pert, &self._cam, mjCAT_ALL, &self._scn)

    def render(self, int viewport_width, int viewport_height):
        cdef mjrRect rect
        rect.left = 0
        rect.bottom = 0
        rect.width = viewport_width
        rect.height = viewport_height
        mjr_render(rect, &self._scn, &self._con)

    def move_camera(self, int action, double reldx, double reldy):
        mjv_moveCamera(self._m, action, reldx, reldy, &self._scn, &self._cam)

    def overlay(self, int font, int gridpos, int viewport_width, int viewport_height, str text1, str text2):
        cdef mjrRect rect
        rect.left = 0
        rect.bottom = 0
        rect.width = viewport_width
        rect.height = viewport_height
        mjr_overlay(font, gridpos, rect, text1.encode(), text2.encode(), &self._con)

    def add_geom(self, int geom_type,
                 np.ndarray[np.float32_t, ndim=1] size,
                 np.ndarray[np.float32_t, ndim=1] pos,
                 np.ndarray[np.float32_t, ndim=1] mat,
                 np.ndarray[np.float32_t, ndim=1] rgba,
                 const char *label):
        # This method in principle doesn't have to be in Cython, but currently
        # the autogenerated wrappers can't handle mjvScene.geoms because its size is determined by
        # mjvScene.maxgeom, which is not a member of mjModel
        if self._closed:
            return
        if self._scn.ngeom >= self._scn.maxgeom:
            raise RuntimeError('Ran out of geoms. maxgeom: %d' % self._scn.maxgeom)
        cdef mjvGeom*g = self._scn.geoms + self._scn.ngeom
        self._scn.ngeom += 1
        g.type = geom_type
        g.dataid = -1
        g.objtype = mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        if label == NULL:
            g.label[0] = 0
        else:
            strncpy(g.label, label, 100)
        for i in range(3): g.size[i] = size[i]
        for i in range(3): g.pos[i] = pos[i]
        for i in range(9): g.mat[i] = mat[i]
        for i in range(4): g.rgba[i] = rgba[i]

from libc.stdlib cimport malloc, free



cdef void euler2quat_nogil(double* vec, double* result) nogil:
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    We can derive this formula in Sympy using:

    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
    '''
    # if len(np.shape(vec)) == 2:
    #     ret = [euler2quat_nogil(vec[i]) for i in range(vec.shape[0])]
    #     return np.stack(ret)
    # else:
    #     assert(vec.shape[0] == 3)
    x, y, z = vec[0], vec[1], vec[2]
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = cmath.cos(z)
    sz = cmath.sin(z)
    cy = cmath.cos(y)
    sy = cmath.sin(y)
    cx = cmath.cos(x)
    sx = cmath.sin(x)
    result[0] = cx*cy*cz - sx*sy*sz
    result[1] = cx*sy*sz + cy*cz*sx
    result[2] = cx*cz*sy - sx*cy*sz
    result[3] = x*cy*sz + sx*cz*sy


cdef class MjParallelLite(object):
    cdef mjModel *_model
    cdef mjData ** data
    cdef mjvPerturb ** perturbation
    cdef int n_parallel
    cdef int num_substeps
    cdef np.ndarray mocap_idxs

    # Exposed in Python for the user's convenience
    cdef readonly PyMjModel model

    cdef readonly tuple site_names, joint_names

    # # cdef readonly PyMjData data
    # cdef readonly tuple body_names, joint_names, geom_names, site_names

    def __cinit__(
            self,
            str xml,
            int n_parallel,
            int num_substeps,
    ):
        logger.debug('mj_loadXML')
        cdef char errstr[300]
        self._model = mj_loadXML(NULL, xml.encode(), errstr, 300)
        if self._model == NULL:
            logger.error('Tried to load this XML: %s' % xml)
            raise Exception('mj_loadXML error: %s' % errstr)
        logger.debug('mj_loadXML completed')
        self.model = WrapMjModel(self._model)

        self.data = <mjData**> malloc(n_parallel * sizeof(mjData*))
        for i in range(n_parallel):
            self.data[i] = mj_makeData(self._model)

        self.perturbation = <mjvPerturb**>malloc(n_parallel * sizeof(mjvPerturb*))
        for i in range(n_parallel):
            self.perturbation[i] = <mjvPerturb*>malloc(sizeof(mjvPerturb))
            mjv_defaultPerturb(self.perturbation[i])
            self.perturbation[i].scale = 1
            self.perturbation[i].active = mjPERT_TRANSLATE | mjPERT_ROTATE

        cdef int[:] body_mocapid = <int[:self._model.nbody]> self._model.body_mocapid

        self.mocap_idxs = np.empty(self._model.nmocap, dtype=np.intc)
        idx = 0
        for i in range(self._model.nbody):
            if body_mocapid[i] == 0:
                self.mocap_idxs[idx] = i
                idx += 1
        assert (idx == self._model.nmocap)


        self.n_parallel = n_parallel
        self.num_substeps = num_substeps
        self.site_names = _extract_mj_names(self._model.names, self._model.name_siteadr, self._model.nsite)
        self.joint_names = _extract_mj_names(self._model.names, self._model.name_jntadr, self._model.njnt)

    def __del__(self):
        for i in range(self.n_parallel):
            mju_free(self.data[i])

        free(self.data)

    @cython.boundscheck(False)
    def forward_dynamics(
            self,
            np.ndarray[double, ndim=2, mode="c"] qpos,
            np.ndarray[double, ndim=2, mode="c"] qvel,
            np.ndarray[double, ndim=2, mode="c"] ctrl,
            np.ndarray[double, ndim=2, mode="c"] mocap,
            int use_site_xpos=False,
            int use_site_xmat=False,
            int use_active_contacts_efc_pos_L2=False,
            int use_qfrc_actuator=False,
            int use_qacc=False,
            int use_qpos=False,
            int use_qvel=False,
            int use_site_jac=False,
            int step=True,
    ):
        cdef int batch_size = qpos.shape[0]
        cdef int i, j
        cdef mjData *data_i
        cdef int dimq = self._model.nq
        cdef int dimv = self._model.nv
        cdef int dimu = self._model.nu
        cdef int dimx = dimq + dimv
        cdef int thread_id
        cdef int dimsite = self._model.nsite
        cdef int nv3
        cdef int site_i
        cdef np.ndarray[double, ndim=2, mode="c"] qpos_next = np.empty((batch_size, dimq))
        cdef np.ndarray[double, ndim=2, mode="c"] qvel_next = np.empty((batch_size, dimv))
        cdef np.ndarray[double, ndim=2, mode="c"] site_xpos
        cdef np.ndarray[double, ndim=2, mode="c"] site_xmat
        cdef np.ndarray[double, ndim=2, mode="c"] qfrc_actuator
        cdef np.ndarray[double, ndim=2, mode="c"] qacc
        cdef np.ndarray[double, ndim=1, mode="c"] active_contacts_efc_pos_L2
        cdef np.ndarray[double, ndim=2, mode="c"] site_jac

        if step and self.model.nmocap > 0:
            assert mocap.shape[mocap.ndim-1] == 6, "mocap should consist of 3 position dimensions, and 3 rotation " \
                                               "dimensions"
            assert qpos.shape[0] == mocap.shape[0]

        cdef double[3] mocap_pos_i
        cdef double[4] mocap_quat_i

        cdef double* jacp
        cdef double* jacr

        sense = dict()

        if use_site_xpos:
            sense["site_xpos"] = site_xpos = np.empty((batch_size, dimsite * 3))
        if use_site_xmat:
            sense["site_xmat"] = site_xmat = np.empty((batch_size, dimsite * 9))
        if use_site_jac:
            sense["site_jac"] = site_jac = np.empty((batch_size, dimv * dimsite * 6))
        if use_qfrc_actuator:
            sense["qfrc_actuator"] = qfrc_actuator = np.empty((batch_size, dimv))
        if use_active_contacts_efc_pos_L2:
            sense["active_contacts_efc_pos_L2"] = active_contacts_efc_pos_L2 = np.empty(batch_size)
        if use_qacc:
            sense["qacc"] = qacc = np.empty((batch_size, dimv))
        if use_qpos:
            sense["qpos"] = qpos_next
        if use_qvel:
            sense["qvel"] = qvel_next

        cdef np.ndarray[int, ndim=1, mode="c"] mocap_idxs = self.mocap_idxs

        with nogil, parallel(num_threads=self.n_parallel):
            for i in prange(batch_size, schedule='static'):
                thread_id = openmp.omp_get_thread_num()
                data_i = self.data[thread_id]
                mju_copy(data_i.qpos, &qpos[i, 0], dimq)
                mju_copy(data_i.qvel, &qvel[i, 0], dimv)

                if step:
                    mju_copy(data_i.ctrl, &ctrl[i, 0], dimu)
                    mju_zero(data_i.xfrc_applied, 6 * self._model.nbody)

                    if self._model.nmocap > 0:
                        for j in range(self._model.nmocap):
                            self.perturbation[thread_id].select = mocap_idxs[j]
                            mju_copy(self.perturbation[thread_id].refpos, &mocap[i, j*6], 3)
                            euler2quat_nogil(&mocap[i, j*6+3], self.perturbation[thread_id].refquat)
                            mjv_applyPerturbPose(self._model, data_i, self.perturbation[thread_id], 0)
                            mjv_applyPerturbForce(self._model, data_i, self.perturbation[thread_id])

                    for j in range(self.num_substeps):
                        mj_step(self._model, data_i)
                mj_forward(self._model, data_i)
                mju_copy(&qpos_next[i, 0], data_i.qpos, dimq)
                mju_copy(&qvel_next[i, 0], data_i.qvel, dimv)
                if use_site_xpos:
                    mju_copy(&site_xpos[i, 0], data_i.site_xpos, dimsite * 3)
                if use_site_xmat:
                    mju_copy(&site_xmat[i, 0], data_i.site_xmat, dimsite * 9)
                if use_qfrc_actuator:
                    mju_copy(&qfrc_actuator[i, 0], data_i.qfrc_actuator, dimv)
                if use_active_contacts_efc_pos_L2:
                    active_contacts_efc_pos_L2[i] = 0
                    for j in range(data_i.ne, data_i.nefc):
                        active_contacts_efc_pos_L2[i] += data_i.efc_pos[j] ** 2
                if use_qacc:
                    mju_copy(&qacc[i, 0], data_i.qacc, dimv)
                if use_site_jac:
                    nv3 = dimv * 3
                    for site_i in range(dimsite):
                        jacp = &site_jac[i, 0] + (site_i * nv3 * 2)
                        jacr = jacp + nv3
                        mj_jacSite(self._model, data_i, jacp, jacr, site_i)

        return np.concatenate([qpos_next, qvel_next], axis=1), sense
