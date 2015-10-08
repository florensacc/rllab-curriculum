from ctypes import Structure, c_int, POINTER, cdll, c_char_p, create_string_buffer
import numpy as np
from numpy.ctypeslib import as_ctypes
import os
from mjtypes import *
from mjlib import mjlib
from util import *

mjLICENSE_OK = 0
mjLICENSE_FILEOPEN = 1
mjLICENSE_FILEREAD = 2
mjLICENSE_INVALID = 3
mjLICENSE_EXPIRED = 4
mjLICENSE_ERROR = 5

class MjError(Exception):
    pass

def register_license(file_path):
    result = mjlib.mj_license(file_path)
    if result == mjLICENSE_OK:
        pass
    elif result == mjLICENSE_FILEOPEN:
        raise MjError('could not open license file')
    elif result == mjLICENSE_FILEREAD:
        raise MjError('could not read data from license file')
    elif result == mjLICENSE_INVALID:
        raise MjError('invalid license')
    elif result == mjLICENSE_EXPIRED:
        raise MjError('expired license')
    elif result == mjLICENSE_ERROR:
        raise MjError('internal error')
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

    def forward(self):
        mjlib.mj_forward(self.ptr, self.data.ptr)

    def step(self):
        mjlib.mj_step(self.ptr, self.data.ptr)

    def __del__(self):
        if self._wrapped is not None:
            mjlib.mj_deleteModel(self._wrapped)


class MjData(MjDataWrapper):

    def __init__(self, wrapped, size_src=None):
        super(MjData, self).__init__(wrapped, size_src)
        
    def __del__(self):
        if self._wrapped is not None:
            mjlib.mj_deleteData(self._wrapped)
