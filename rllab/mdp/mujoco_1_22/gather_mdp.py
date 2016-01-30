from ant_mdp import AntMDP
from rllab.misc.overrides import overrides
from ctypes import pointer, byref
from rllab.mjcapi.rocky_mjc_1_22 import MjViewer, MjModel, mjcore, mjlib, mjextra
from gather.cube_viewer import CubeViewer
import os.path as osp


class GatherViewer(MjViewer):

    def __init__(self):
        super(GatherViewer, self).__init__()
        self.cube_renderer = CubeViewer()
        cube_model = MjModel(osp.abspath(
            osp.join(
                osp.dirname(__file__),
                '../../../vendor/mujoco_models/1_22/box.xml'
            )
        ))
        self.cube_renderer.set_model(cube_model)

    def start(self):
        super(GatherViewer, self).start()
        self.cube_renderer.start(self.window)

    def handle_mouse_move(self, window, xpos, ypos):
        super(GatherViewer, self).handle_mouse_move(window, xpos, ypos)
        self.cube_renderer.handle_mouse_move(window, xpos, ypos)

    def handle_scroll(self, window, x_offset, y_offset):
        super(GatherViewer, self).handle_scroll(window, x_offset, y_offset)
        self.cube_renderer.handle_scroll(window, x_offset, y_offset)

    def render(self):
        super(GatherViewer, self).render()
        self.cube_renderer.render()
        tmpobjects = mjcore.MJVOBJECTS()
        mjlib.mjlib.mjv_makeObjects(byref(tmpobjects), 1000)
        mjextra.append_objects(tmpobjects, self.cube_renderer.objects)
        mjextra.append_objects(tmpobjects, self.objects)
        mjlib.mjlib.mjv_makeLights(self.model.ptr, self.data.ptr, byref(tmpobjects))
        mjlib.mjlib.mjr_render(0, self.get_rect(), byref(tmpobjects), byref(self.ropt), byref(self.cam.pose), byref(self.con))


class GatherMDP(AntMDP):

    @overrides
    def get_viewer(self):
        if self.viewer is None:
            self.viewer = GatherViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer
