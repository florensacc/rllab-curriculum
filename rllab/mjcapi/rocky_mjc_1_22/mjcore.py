from ctypes import Structure, c_int, POINTER, cdll, c_char_p, create_string_buffer
import numpy as np
from numpy.ctypeslib import as_ctypes
import os
from mjtypes import *
from mjlib import mjlib
from util import *


class MjError(Exception):
    pass

def register_license(file_path):
    result = mjlib.mj_activate(file_path)
    if result == 1:
        pass
    elif result == 0:
        raise MjError('could not register license')
    else:
        raise MjError("I don't know wth happene")

class MjModel(MjModelWrapper):

    def __init__(self, xml_path):
        buf = create_string_buffer(300)
        model_ptr = mjlib.mj_loadXML(xml_path, buf)
        if len(buf.value) > 0:
            super(MjModel, self).__init__(None)
            raise MjError(buf.value)
        super(MjModel, self).__init__(model_ptr)
        data_ptr = mjlib.mj_makeData(model_ptr)
        data = MjData(data_ptr, self)
        self.data = data
        self.forward()

    # This is like updating the state of mujoco. I'm not really sure what it's updating though
    def forward(self):
        mjlib.mj_forward(self.ptr, self.data.ptr)

    def step(self):
        mjlib.mj_step(self.ptr, self.data.ptr)

    def __del__(self):
        if self._wrapped is not None:
            mjlib.mj_deleteModel(self._wrapped)

    @property
    def body_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_bodyadr.flatten()]

    @property
    def joint_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_jntadr.flatten()]

    @property
    def geom_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_geomadr.flatten()]

    @property
    def site_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_siteadr.flatten()]

    @property
    def mesh_names(self):
        start_addr = ctypes.addressof(self.names.contents)
        return [ctypes.string_at(start_addr + inc)
                for inc in self.name_meshadr.flatten()]


class MjData(MjDataWrapper):

    def __init__(self, wrapped, size_src=None):
        super(MjData, self).__init__(wrapped, size_src)
        
    def __del__(self):
        if self._wrapped is not None:
            mjlib.mj_deleteData(self._wrapped)
