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


# _FLOAT_EPS_4 = np.finfo(np.float_).eps * 4.0
#
#
#
# cdef np.ndarray[double, ndim=2] euler2mat(np.ndarray[double, ndim=1] euler) nogil:
#     ''' Return matrix for rotations around z, y and x axes
#     Uses the z, then y, then x convention above
#     Parameters
#     ----------
#     z : scalar
#        Rotation angle in radians around z-axis (performed first)
#     y : scalar
#        Rotation angle in radians around y-axis
#     x : scalar
#        Rotation angle in radians around x-axis (performed last)
#     Returns
#     -------
#     M : array shape (3,3)
#        Rotation matrix giving same rotation as for given angles
#     Examples
#     --------
#     >>> zrot = 1.3 # radians
#     >>> yrot = -0.1
#     >>> xrot = 0.2
#     >>> M = euler2mat(zrot, yrot, xrot)
#     >>> M.shape == (3, 3)
#     True
#     The output rotation matrix is equal to the composition of the
#     individual rotations
#     >>> M1 = euler2mat(zrot)
#     >>> M2 = euler2mat(0, yrot)
#     >>> M3 = euler2mat(0, 0, xrot)
#     >>> composed_M = np.dot(M3, np.dot(M2, M1))
#     >>> np.allclose(M, composed_M)
#     True
#     You can specify rotations by named arguments
#     >>> np.all(M3 == euler2mat(x=xrot))
#     True
#     When applying M to a vector, the vector should column vector to the
#     right of M.  If the right hand side is a 2D array rather than a
#     vector, then each column of the 2D array represents a vector.
#     >>> vec = np.array([1, 0, 0]).reshape((3,1))
#     >>> v2 = np.dot(M, vec)
#     >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
#     >>> vecs2 = np.dot(M, vecs)
#     Rotations are counter-clockwise.
#     >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
#     >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
#     True
#     >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
#     >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
#     True
#     >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
#     >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
#     True
#     Notes
#     -----
#     The direction of rotation is given by the right-hand rule (orient
#     the thumb of the right hand along the axis around which the rotation
#     occurs, with the end of the thumb at the positive end of the axis;
#     curl your fingers; the direction your fingers curl is the direction
#     of rotation).  Therefore, the rotations are counterclockwise if
#     looking along the axis of rotation from positive to negative.
#     '''
#     cdef double z = euler[0]
#     cdef double y = euler[1]
#     cdef double x = euler[2]
#     cdef np.ndarray[double, ndim=2] ret = np.eye(3)#(3, 3))
#     # ret[0, 0] = 1
#     # ret[1, 1] = 1
#     # ret[2, 2] = 1
#     # Ms = []
#     if z:
#         cosz = math.cos(z)
#         sinz = math.sin(z)
#         ret = np.dot(np.array([[cosz, -sinz, 0],
#                             [sinz, cosz, 0],
#                             [0, 0, 1]]), ret)
#     if y:
#         cosy = math.cos(y)
#         siny = math.sin(y)
#         ret = np.dot(np.array([[cosy, 0, siny],
#                             [0, 1, 0],
#                             [-siny, 0, cosy]]), ret)
#     if x:
#         cosx = math.cos(x)
#         sinx = math.sin(x)
#         ret = np.dot(np.array([[1, 0, 0],
#                             [0, cosx, -sinx],
#                             [0, sinx, cosx]]), ret)
#     return ret
#
#
# cdef np.ndarray[double, ndim=2] mat2euler(np.ndarray[double, ndim=2] M) nogil:
#     ''' Discover Euler angle vector from 3x3 matrix
#     Uses the conventions above.
#     Parameters
#     ----------
#     M : array-like, shape (3,3)
#     cy_thresh : None or scalar, optional
#        threshold below which to give up on straightforward arctan for
#        estimating x rotation.  If None (default), estimate from
#        precision of input.
#     Returns
#     -------
#     z : scalar
#     y : scalar
#     x : scalar
#        Rotations in radians around z, y, x axes, respectively
#     Notes
#     -----
#     If there was no numerical error, the routine could be derived using
#     Sympy expression for z then y then x rotation matrix, which is::
#       [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
#       [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
#       [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
#     with the obvious derivations for z, y, and x
#        z = atan2(-r12, r11)
#        y = asin(r13)
#        x = atan2(-r23, r33)
#     Problems arise when cos(y) is close to zero, because both of::
#        z = atan2(cos(y)*sin(z), cos(y)*cos(z))
#        x = atan2(cos(y)*sin(x), cos(x)*cos(y))
#     will be close to atan2(0, 0), and highly unstable.
#     The ``cy`` fix for numerical instability below is from: *Graphics
#     Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
#     0123361559.  Specifically it comes from EulerAngles.c by Ken
#     Shoemake, and deals with the case where cos(y) is close to zero:
#     See: http://www.graphicsgems.org/
#     The code appears to be licensed (from the website) as "can be used
#     without restrictions".
#     '''
#     # M = np.asarray(M)
#     if cy_thresh is None:
#         # try:
#         #     cy_thresh = np.finfo(M.dtype).eps * 4
#         # except ValueError:
#         cy_thresh = _FLOAT_EPS_4
#     r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
#     # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
#     cy = cmath.sqrt(r33 * r33 + r23 * r23)
#     if cy > cy_thresh:  # cos(y) not close to zero, standard form
#         z = cmath.atan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
#         y = cmath.atan2(r13, cy)  # atan2(sin(y), cy)
#         x = cmath.atan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
#     else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
#         # so r21 -> sin(z), r22 -> cos(z) and
#         z = cmath.atan2(r21, r22)
#         y = cmath.atan2(r13, cy)  # atan2(sin(y), cy)
#         x = 0.0
#     return z, y, x


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

    cdef str obs_type
    # # cdef readonly PyMjData data
    # cdef readonly tuple body_names, joint_names, geom_names, site_names

    def __cinit__(
            self,
            str xml,
            int n_parallel,
            int num_substeps,
            str obs_type,
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
        self.obs_type = obs_type

    def __del__(self):
        for i in range(self.n_parallel):
            mju_free(self.data[i])

        free(self.data)
    # cdef get_relative_frame_i(self):
    #     # TODO: cache
    #     gripper_xpos = np.empty((x.shape[0], 3))
    #     gripper_xmat = np.empty((x.shape[0], 3, 3))
    #
    #     def set_gripper_pos(sense, i):
    #         gripper_xpos[i] = sense["site_stall_mocap_xpos"]
    #         gripper_xmat[i] = sense["site_stall_mocap_xmat"].reshape(3, 3)
    #
    #     self.model.forward(x[:, :self.dimq], x[:, self.dimq:], lambda_over_sense=set_gripper_pos)
    #
    #     import ipdb; ipdb.set_trace()
    #
    #     gripper_xpos = gripper_xpos.reshape(-1, 3)
    #     gripper_xmat = gripper_xmat.reshape(-1, 3, 3)
    #
    #     return gripper_xpos, gripper_xmat

    # cdef preprocess_mocap(self, mjData* data_i, double* mocap) nogil:
    #     # mocapdim = mocap.shape[0] / 2
    #     # mocap_xpos = mocap[..., :mocapdim]
    #     # mocap_euler = mocap[..., mocapdim:]
    #
    #     cdef int site_idx = list(self._model.site_names).index("stall_mocap")
    #     cdef int i
    #
    #     if self.obs_type == "relative":
    #         gripper_xpos = data_i.site_xpos[site_idx*3:site_idx*3+3]
    #         gripper_xmat = data_i.site_xmat[site_idx*9:site_idx*9+9].reshape((3, 3))
    #
    #         for i in range(3): # mocap_xpos
    #             mocap[i] *= self.mocap_move_speed
    #         for i in range(3, 6): # mocap_euler
    #             mocap[i] *= self.mocap_rot_speed
    #
    #         # mocap_xpos *= self.mocap_move_speed
    #         # mocap_euler *= self.mocap_rot_speed
    #
    #         mocap_xpos[:] = gripper_xpos + np.matmul(gripper_xmat, mocap_xpos.reshape(3, 1)).reshape(3)
    #         mocap_euler[:] = mat2euler(np.matmul(gripper_xmat, euler2mat(mocap_euler)))
    #     elif self.obs_type == "flatten":
    #         mocap_xpos[:] = [0.3, 0.0, 0.7] + [0.15, 0.2, 0.1] * mocap_xpos  # TODO: refactor FIXME
    #
    #     if self.params.mocap_fix_orientation:
    #         mocap_euler[0] = 0
    #         mocap_euler[1] = 0.5 * cmath.pi
    #         mocap_euler[2] = 0

        # mocap = np.concatenate((mocap_xpos, mocap_euler), axis=1)
        #
        # return mocap

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

        # mocap = self.preprocess_mocap(x, mocap)

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
                    # mju_copy(&site_jac[i, 0], data_i.site)

        return np.concatenate([qpos_next, qvel_next], axis=1), sense
