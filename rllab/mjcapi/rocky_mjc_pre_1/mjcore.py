from __future__ import absolute_import
import ctypes
from rllab.mjcapi.rocky_mjc_pre_2.mjtypes import MjModelWrapper, \
    MjDataWrapper, MjOptionWrapper
from rllab.mjcapi.rocky_mjc_pre_2.mjlib import mjlib, wrapper_lib


class MjError(Exception):
    pass


class MjModel(MjModelWrapper):

    def __init__(self, xml_path):
        buf = ctypes.create_string_buffer(300)
        model_ptr = wrapper_lib.mj_loadXML(xml_path, buf)
        if len(buf.value) > 0:
            super(MjModel, self).__init__(None)
            raise MjError(buf.value)
        super(MjModel, self).__init__(model_ptr)
        data_ptr = mjlib.mj_makeData(model_ptr)
        data = MjData(data_ptr, self)
        option_ptr = wrapper_lib.mj_loadXMLOption(xml_path, buf)
        if len(buf.value) > 0:
            raise MjError(buf.value)
        option = MjOption(option_ptr, self)

        self.data = data
        self.option = option
        self.forward()

    def forward(self):
        mjlib.mj_forward(self.ptr, self.option.ptr, self.data.ptr)

    def step(self):
        mjlib.mj_step(self.ptr, self.option.ptr, self.data.ptr)

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


class MjOption(MjOptionWrapper):

    def __init__(self, wrapped, size_src=None):
        super(MjOption, self).__init__(wrapped, size_src)

    def __del__(self):
        if self._wrapped is not None:
            mjlib.mj_deleteOption(self._wrapped)
