import mjcore
import os
import numpy as np
from mjlib import mjlib
from ctypes import pointer

xml_path = "../vendor/mujoco_osx/humanoid.xml"


#print model.body_inertia

objects = mjcore.MJVOBJECTS()
cam = mjcore.MJVCAMERA()
vopt = mjcore.MJVOPTION()
ropt = mjcore.MJROPTION()
con = mjcore.MJRCONTEXT()


mjlib.mjv_makeObjects(pointer(mjvObject), 1000)
mjlib.mjv_defaultCamera(pointer(cam));
mjlib.mjv_defaultOption(pointer(vopt));
mjlib.mjr_defaultOption(pointer(ropt));
mjlib.mjr_defaultContext(pointer(con));

model, data = mjcore.load_model_data(xml_path)

mjlib.mjr_makeContext(model.ptr, pointer(con), 150);

#mjvObject
#mujoco_lib.mjvObject
