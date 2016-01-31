# from .mujoco_mdp import MujocoMDP
# from .ant_mdp import AntMDP
from .swimmer_mdp import SwimmerMDP
from rllab.misc.overrides import overrides
from ctypes import byref
from rllab.mjcapi.rocky_mjc_1_22 import MjViewer, MjModel, mjcore, mjlib, mjextra
from gather.cube_viewer import CubeViewer
import os.path as osp
import numpy as np
from rllab.core.serializable import Serializable

FOOD = 0
BOMB = 1


class GatherViewer(MjViewer):

    def __init__(self):
        super(GatherViewer, self).__init__()
        self.cube_renderer = CubeViewer()
        cube_model = MjModel(osp.abspath(
            osp.join(
                osp.dirname(__file__),
                '../../../vendor/mujoco_models/1_22/ball.xml'
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
        mjlib.mjlib.mjv_makeLights(
            self.model.ptr, self.data.ptr, byref(tmpobjects))
        mjlib.mjlib.mjr_render(0, self.get_rect(), byref(tmpobjects), byref(
            self.ropt), byref(self.cam.pose), byref(self.con))


class GatherMDP(SwimmerMDP, Serializable):

    FILE = "swimmer_gather.xml"

    def __init__(
            self,
            n_items=50,
            activity_range=28,
            *args, **kwargs
    ):
        super(GatherMDP, self).__init__(*args, **kwargs)
        self.n_items = n_items
        self.activity_range = activity_range
        self.reset()
        Serializable.quick_init(self, locals())

    def reset(self):
        super(GatherMDP, self).reset()
        self.object_list = []
        for _ in xrange(self.n_items):
            x = np.random.randint(-self.activity_range, self.activity_range)
            y = np.random.randint(-self.activity_range, self.activity_range)
            typ = np.random.choice([FOOD, BOMB])
            self.object_list.append((x, y, typ))

    @overrides
    def get_viewer(self):
        if self.viewer is None:
            self.viewer = GatherViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer
