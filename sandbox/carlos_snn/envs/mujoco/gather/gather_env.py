import math
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
from ctypes import byref
from rllab.misc import logger

import numpy as np
import theano

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.env_spec import EnvSpec
from rllab.envs.base import Env, Step
from rllab.envs.mujoco.gather.embedded_viewer import EmbeddedViewer
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.mujoco_py import MjViewer, MjModel, mjcore, mjlib, \
    mjextra, glfw
from rllab.envs.mujoco.gather.gather_env import GatherViewer

APPLE = 0
BOMB = 1


class GatherEnv(ProxyEnv, Serializable):
    MODEL_CLASS = None
    ORI_IND = None

    @autoargs.arg('n_apples', type=int,
                  help='Number of apples in each episode')
    @autoargs.arg('n_bombs', type=int,
                  help='Number of bombs in each episode')
    @autoargs.arg('activity_range', type=float,
                  help='The span for generating objects '
                       '(x, y in [-range, range])')
    @autoargs.arg('robot_object_spacing', type=float,
                  help='Number of objects in each episode')
    @autoargs.arg('catch_range', type=float,
                  help='Minimum distance range to catch an object')
    @autoargs.arg('n_bins', type=float,
                  help='Number of objects in each episode')
    @autoargs.arg('sensor_range', type=float,
                  help='Maximum sensor range (how far it can go)')
    @autoargs.arg('sensor_span', type=float,
                  help='Maximum sensor span (how wide it can span), in '
                       'radians')
    def __init__(
            self,
            n_apples=8,
            n_bombs=8,
            activity_range=6.,
            robot_object_spacing=2.,
            catch_range=1.,
            n_bins=10,
            sensor_range=6.,
            sensor_span=math.pi,
            coef_inner_rew=0.,
            dying_cost=-10,
            *args, **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.n_apples = n_apples
        self.n_bombs = n_bombs
        self.activity_range = activity_range
        self.robot_object_spacing = robot_object_spacing
        self.catch_range = catch_range
        self.n_bins = n_bins
        self.sensor_range = sensor_range
        self.sensor_span = sensor_span
        self.coef_inner_rew = coef_inner_rew
        self.dying_cost = dying_cost
        self.objects = []
        self.viewer = None
        # super(GatherEnv, self).__init__(*args, **kwargs)
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")
        attrs = dict(
            type="box", conaffinity="1", rgba="0.8 0.9 0.8 1", condim="3"
        )
        walldist = self.activity_range + 1
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall1",
                pos="0 -%d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall2",
                pos="0 %d 0" % walldist,
                size="%d.5 0.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall3",
                pos="-%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="wall4",
                pos="%d 0 0" % walldist,
                size="0.5 %d.5 1" % walldist))
        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)
        # pylint: disable=not-callable
        inner_env = model_cls(*args, file_path=file_path, **kwargs)
        # pylint: enable=not-callable
        # import pdb; pdb.set_trace()
        ProxyEnv.__init__(self, inner_env)

    def reset(self, also_wrapped=True):
        # super(GatherMDP, self).reset()
        self.objects = []
        existing = set()
        while len(self.objects) < self.n_apples:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = APPLE
            self.objects.append((x, y, typ))
            existing.add((x, y))
        while len(self.objects) < self.n_apples + self.n_bombs:
            x = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            y = np.random.randint(-self.activity_range / 2,
                                  self.activity_range / 2) * 2
            # regenerate, since it is too close to the robot's initial position
            if x ** 2 + y ** 2 < self.robot_object_spacing ** 2:
                continue
            if (x, y) in existing:
                continue
            typ = BOMB
            self.objects.append((x, y, typ))
            existing.add((x, y))

        if also_wrapped:  # CF
            self.wrapped_env.reset()
        return self.get_current_obs()

    def step(self, action):
        _, inner_rew, done, info = self.wrapped_env.step(action)
        info['inner_rewards'] = inner_rew
        info['gather_rewards'] = 0
        if done:
            return Step(self.get_current_obs(), self.dying_cost, done, **info)  # give a -10 rew if the robot dies
        com = self.wrapped_env.get_body_com("torso")
        x, y = com[:2]
        reward = self.coef_inner_rew * inner_rew
        new_objs = []
        for obj in self.objects:
            ox, oy, typ = obj
            # object within zone!
            if (ox - x) ** 2 + (oy - y) ** 2 < self.catch_range ** 2:
                if typ == APPLE:
                    reward = reward + 1
                    info['gather_rewards'] = 1
                else:
                    reward = reward - 1
                    info['gather_rewards'] = -1
            else:
                new_objs.append(obj)
        self.objects = new_objs
        done = len(self.objects) == 0
        return Step(self.get_current_obs(), reward, done, **info)

    def get_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.wrapped_env.get_body_com("torso")[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        sorted_objects = sorted(
            self.objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.get_ori()  # overwrite this for Ant!

        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                import ipdb
                ipdb.set_trace()
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            #    ((angle + half_span) +
            #     ori) % (2 * math.pi) / bin_res)
            intensity = 1.0 - dist / self.sensor_range
            if typ == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity
        return apple_readings, bomb_readings   # in maze this is given already concatenated!! @@@@@@@

    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()

    def get_current_obs(self):
        # return sensor data along with data about itself
        self_obs = self.wrapped_env.get_current_obs()
        apple_readings, bomb_readings = self.get_readings()
        return np.concatenate([self_obs, apple_readings, bomb_readings])

    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    # CF space of only the robot observations (they go first in the get current obs)
    @property
    def robot_observation_space(self):
        shp = self.get_current_robot_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    def maze_observation_space(self):
        shp = np.concatenate(self.get_readings()).shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    @overrides
    def action_space(self):
        return self.wrapped_env.action_space

    @property
    def action_bounds(self):
        return self.wrapped_env.action_bounds

    # @property
    # def viewer(self):
    #     return self.wrapped_env.viewer

    def action_from_key(self, key):
        return self.wrapped_env.action_from_key(key)

    def get_viewer(self):
        if self.wrapped_env.viewer is None:
            self.wrapped_env.viewer = GatherViewer(self)
            self.wrapped_env.viewer.start()
            self.wrapped_env.viewer.set_model(self.wrapped_env.model)
        return self.wrapped_env.viewer

    def stop_viewer(self):
        if self.wrapped_env.viewer:
            self.wrapped_env.viewer.finish()

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            self.get_viewer().render()
            data, width, height = self.get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        elif mode == 'human':
            # self.get_viewer().loop_once()
            self.get_viewer()
            self.wrapped_env.render()
        if close:
            self.stop_viewer()

    def get_ori(self):
        return self.wrapped_env.model.data.qpos[self.__class__.ORI_IND]

    @overrides
    def log_diagnostics(self, paths, log_prefix='Gather', *args, **kwargs):
        # we call here any logging related to the gather, strip the maze obs and call log_diag with the stripped paths
        # we need to log the purely gather reward!!
        with logger.tabular_prefix(log_prefix + '_'):
            gather_undiscounted_returns = [sum(path['env_infos']['gather_rewards']) for path in paths]
            logger.record_tabular_misc_stat('Return', gather_undiscounted_returns, placement='front')
        stripped_paths = []
        for path in paths:
            stripped_path = {}
            for k, v in path.items():
                stripped_path[k] = v
            stripped_path['observations'] = \
                stripped_path['observations'][:, :self.wrapped_env.observation_space.flat_dim]
            #  this breaks if the obs of the robot are d>1 dimensional (not a vector)
            stripped_paths.append(stripped_path)
        with logger.tabular_prefix('wrapped_'):
            if 'env_infos' in paths[0].keys() and 'inner_reward' in paths[0]['env_infos'].keys():
                wrapped_undiscounted_return = np.mean([np.sum(path['env_infos']['inner_reward']) for path in paths])
                logger.record_tabular('AverageReturn', wrapped_undiscounted_return)
            self.wrapped_env.log_diagnostics(stripped_paths)  # see swimmer_env.py for a scketch of the maze plotting!



