from ctypes import *
import os
from util import *
from mjtypes import *

osp = os.path
if sys.platform.startswith("darwin"):
    libfile = osp.abspath(osp.join(
        osp.dirname(__file__),
        "../../../vendor/mujoco_pre_2/osx/libmjc2.dylib"))
    wrapper_libfile = osp.abspath(osp.join(
        osp.dirname(__file__),
        "../../../vendor/mujoco_pre_2/osx/libmjc2_wrapper.dylib"))
    osg_libfile = osp.abspath(osp.join(
        osp.dirname(__file__),
        "../../../vendor/mujoco_pre_2/osx/libmjc2_osg.dylib"))
elif sys.platform.startswith("linux"):
    raise NotImplementedError
elif sys.platform.startswith("win"):
    raise NotImplementedError
else:
    raise RuntimeError("unrecognized platform %s" % sys.platform)


mjlib = cdll.LoadLibrary(libfile)
wrapper_lib = cdll.LoadLibrary(wrapper_libfile)
osg_lib = cdll.LoadLibrary(osg_libfile)


wrapper_lib.mj_loadXML.argtypes = [c_char_p, c_char_p]
wrapper_lib.mj_loadXML.restype = POINTER(MJMODEL)
wrapper_lib.mj_loadXMLOption.argtypes = [c_char_p, c_char_p]
wrapper_lib.mj_loadXMLOption.restype = POINTER(MJOPTION)

osg_lib.viewer_new.restype = c_void_p
osg_lib.SetModel.argtypes = [c_void_p, POINTER(MJMODEL)]
osg_lib.SetData.argtypes = [c_void_p, POINTER(MJDATA)]
osg_lib.Idle.argtypes = [c_void_p]
osg_lib.RenderOnce.argtypes = [c_void_p]

mjlib.mj_step.argtypes = [POINTER(MJMODEL), POINTER(MJOPTION), POINTER(MJDATA)]
mjlib.mj_step.restype = None

mjlib.mj_deleteOption.argtypes = [POINTER(MJOPTION)]
mjlib.mj_step.restype = None

mjlib.mj_forward.argtypes = [POINTER(MJMODEL), POINTER(MJOPTION), POINTER(MJDATA)]
mjlib.mj_forward.restype = None
# mjlib.mj_inverse.argtypes = [POINTER(MJMODEL), POINTER(MJDATA)]
# mjlib.mj_inverse.restype = None
# 
# mjlib.mj_deleteModel.argtypes = [POINTER(MJMODEL)]
# mjlib.mj_deleteModel.restype = None
# 
mjlib.mj_makeData.argtypes = [POINTER(MJMODEL)]
mjlib.mj_makeData.restype = POINTER(MJDATA)
# 
mjlib.mj_deleteData.argtypes = [POINTER(MJDATA)]
mjlib.mj_deleteData.restype = None
# 
# mjlib.mjv_makeObjects.argtypes = [POINTER(MJVOBJECTS), c_int]
# mjlib.mjv_makeObjects.restype = None
# 
# mjlib.mjv_freeObjects.argtypes = [POINTER(MJVOBJECTS)]
# mjlib.mjv_freeObjects.restype = None
# mjlib.mjv_defaultOption.argtypes = [POINTER(MJVOPTION)]
# mjlib.mjv_defaultOption.restype = None
# 
# mjlib.mjv_defaultCamera.argtypes = [POINTER(MJVCAMERA)]
# mjlib.mjv_defaultCamera.restype = None
# mjlib.mjv_setCamera.argtypes = [POINTER(MJMODEL), POINTER(MJDATA), POINTER(MJVCAMERA)]
# mjlib.mjv_setCamera.restype = None
# mjlib.mjv_updateCameraPose.argtypes = [POINTER(MJVCAMERA), c_double]
# mjlib.mjv_updateCameraPose.restype = None
# #mjlib.mjv_convert3D.argtypes = [POINTER(c_double), POINTER(c_double), c_double, POINTER(MJVCAMERAPOSE)]
# #mjlib.mjv_convert3D.restype = None
# #mjlib.mjv_convert2D.argtypes = [POINTER(c_double), mjtMouse, c_double, c_double, c_double, POINTER(MJVCAMERAPOSE)]
# #mjlib.mjv_convert2D.restype = None
# mjlib.mjv_moveCamera.argtypes = [c_int, c_float, c_float, POINTER(MJVCAMERA), c_float, c_float]
# mjlib.mjv_moveCamera.restype = None
# #mjlib.mjv_moveObject.argtypes = [mjtMouse, c_float, c_float, POINTER(MJVCAMERAPOSE), c_float, c_float, POINTER(c_double), POINTER(c_double)]
# #mjlib.mjv_moveObject.restype = None
# #mjlib.mjv_mousePerturb.argtypes = [POINTER(MJMODEL), POINTER(MJDATA), c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
# #mjlib.mjv_mousePerturb.restype = None
# #mjlib.mjv_mouseEdit.argtypes = [POINTER(MJMODEL), POINTER(MJDATA), c_int, c_int, POINTER(c_double), POINTER(c_double)]
# #mjlib.mjv_mouseEdit.restype = None
# mjlib.mjv_makeGeoms.argtypes = [POINTER(MJMODEL), POINTER(MJDATA), POINTER(MJVOBJECTS), POINTER(MJVOPTION), c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
# mjlib.mjv_makeGeoms.restype = None
# mjlib.mjv_makeLights.argtypes = [POINTER(MJMODEL), POINTER(MJDATA), POINTER(MJVOBJECTS)]
# mjlib.mjv_makeLights.restype = None
# mjlib.mjr_overlay.argtypes = [MJRRECT, c_int, c_int, String, String, POINTER(MJRCONTEXT)]
# mjlib.mjr_overlay.restype = None
# #mjlib.mjr_rectangle.argtypes = [c_int, MJRRECT, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double]
# #mjlib.mjr_rectangle.restype = None
# #mjlib.mjr_finish.argtypes = []
# #mjlib.mjr_finish.restype = None
# #mjlib.mjr_text.argtypes = [String, POINTER(MJRCONTEXT), c_int, c_float, c_float, c_float, c_float, c_float, c_float]
# #mjlib.mjr_text.restype = None
# #mjlib.mjr_textback.argtypes = [String, POINTER(MJRCONTEXT), c_float, c_float, c_float, c_float, c_float, c_float]
# #mjlib.mjr_textback.restype = None
# #mjlib.mjr_textWidth.argtypes = [String, POINTER(MJRCONTEXT), c_int]
# #mjlib.mjr_textWidth.restype = c_int
# mjlib.mjr_defaultOption.argtypes = [POINTER(MJROPTION)]
# mjlib.mjr_defaultOption.restype = None
# mjlib.mjr_defaultContext.argtypes = [POINTER(MJRCONTEXT)]
# mjlib.mjr_defaultContext.restype = None
# #mjlib.mjr_uploadTexture.argtypes = [POINTER(MJMODEL), POINTER(MJRCONTEXT), c_int]
# #mjlib.mjr_uploadTexture.restype = None
# mjlib.mjr_makeContext.argtypes = [POINTER(MJMODEL), POINTER(MJRCONTEXT), c_int]
# mjlib.mjr_makeContext.restype = None
# mjlib.mjr_freeContext.argtypes = [POINTER(MJRCONTEXT)]
# mjlib.mjr_freeContext.restype = None
# mjlib.mjr_render.argtypes = [c_int, MJRRECT, POINTER(MJVOBJECTS), POINTER(MJROPTION), POINTER(MJVCAMERAPOSE), POINTER(MJRCONTEXT)]
# mjlib.mjr_render.restype = None
