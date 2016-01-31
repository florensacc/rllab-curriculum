# from .mujoco_mdp import MujocoMDP
from .ant_mdp import AntMDP
# from .swimmer_mdp import SwimmerMDP
from rllab.misc.overrides import overrides
from ctypes import byref
from rllab.mjcapi.rocky_mjc_1_22 import MjViewer, MjModel, mjcore, mjlib, mjextra
from gather.embedded_viewer import EmbeddedViewer
import os.path as osp
import numpy as np
from rllab.core.serializable import Serializable

FOOD = 0
BOMB = 1


class GatherViewer(MjViewer):

    def __init__(self, mdp):
        self.mdp = mdp
        super(GatherViewer, self).__init__()
        green_ball_model = MjModel(osp.abspath(
            osp.join(
                osp.dirname(__file__),
                '../../../vendor/mujoco_models/1_22/green_ball.xml'
            )
        ))
        self.green_ball_renderer = EmbeddedViewer()
        self.green_ball_model = green_ball_model
        self.green_ball_renderer.set_model(green_ball_model)
        red_ball_model = MjModel(osp.abspath(
            osp.join(
                osp.dirname(__file__),
                '../../../vendor/mujoco_models/1_22/red_ball.xml'
            )
        ))
        self.red_ball_renderer = EmbeddedViewer()
        self.red_ball_model = red_ball_model
        self.red_ball_renderer.set_model(red_ball_model)
        
    def start(self):
        super(GatherViewer, self).start()
        self.green_ball_renderer.start(self.window)
        self.red_ball_renderer.start(self.window)

    def handle_mouse_move(self, window, xpos, ypos):
        super(GatherViewer, self).handle_mouse_move(window, xpos, ypos)
        self.green_ball_renderer.handle_mouse_move(window, xpos, ypos)
        self.red_ball_renderer.handle_mouse_move(window, xpos, ypos)

    def handle_scroll(self, window, x_offset, y_offset):
        super(GatherViewer, self).handle_scroll(window, x_offset, y_offset)
        self.green_ball_renderer.handle_scroll(window, x_offset, y_offset)
        self.red_ball_renderer.handle_scroll(window, x_offset, y_offset)

    def render(self):
        super(GatherViewer, self).render()
        tmpobjects = mjcore.MJVOBJECTS()
        mjlib.mjlib.mjv_makeObjects(byref(tmpobjects), 1000)
        for obj in self.mdp.objects:
            x, y, typ = obj
            # print x, y
            qpos = np.zeros_like(self.green_ball_model.data.qpos)
            qpos[0, 0] = x
            qpos[1, 0] = y
            if typ == FOOD:
                self.green_ball_model.data.qpos = qpos
                self.green_ball_model.forward()
                self.green_ball_renderer.render()
                mjextra.append_objects(
                    tmpobjects, self.green_ball_renderer.objects)
            else:
                self.red_ball_model.data.qpos = qpos
                self.red_ball_model.forward()
                self.red_ball_renderer.render()
                mjextra.append_objects(
                    tmpobjects, self.red_ball_renderer.objects)
        mjextra.append_objects(tmpobjects, self.objects)
        mjlib.mjlib.mjv_makeLights(
            self.model.ptr, self.data.ptr, byref(tmpobjects))
        mjlib.mjlib.mjr_render(0, self.get_rect(), byref(tmpobjects), byref(
            self.ropt), byref(self.cam.pose), byref(self.con))


class GatherMDP(AntMDP, Serializable):

    FILE = "ant_gather.xml"

    def __init__(
            self,
            n_items=20,
            activity_range=6,
            robot_object_spacing=3,
            object_object_spacing=3,
            *args, **kwargs
    ):
        self.n_items = n_items
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.object_object_spacing = object_object_spacing
        super(GatherMDP, self).__init__(*args, **kwargs)
        self.reset()
        Serializable.quick_init(self, locals())

    def reset(self):
        ret = super(GatherMDP, self).reset()
        self.objects = []
        while len(self.objects) < self.n_items:
            # for _ in xrange(self.n_items):
            x = np.random.randint(-self.activity_range, self.activity_range)
            y = np.random.randint(-self.activity_range, self.activity_range)
            if x**2+y**2 < self.robot_object_spacing**2:
                continue
            #satisfied = True
            #for obj in self.objects:
            #    ox, oy, _ = obj
                #if (ox-x)**2 + (oy-y)**2 < self.object_object_spacing**2:
                #    satisfied = False
                #    break
            #if not satisfied:
            #    continue
            typ = np.random.choice([FOOD, BOMB])
            self.objects.append((x, y, typ))
        return ret

    def step(self, state, action):
        next_state, next_obs, reward, done = \
            super(GatherMDP, self).step(state, action)
        if done:
            return next_state, self.get_current_obs(), 0, done
        com = self.get_body_com("torso")
        x, y = com[:2]
        new_objs = []
        for obj in self.objects:
            ox, oy, _ = obj
            if (ox-x)**2 + (oy-y)**2 < 1**2:
                pass
            else:
                new_objs.append(obj)
        self.objects = new_objs
        return next_state, self.get_current_obs(), 0, done

    @overrides
    def get_viewer(self):
        if self.viewer is None:
            self.viewer = GatherViewer(self)
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer
