import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np

from rllab import spaces
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
# from rllab.envs.mujoco.maze.maze_env_utils import construct_maze
from curriculum.envs.maze.maze_env_utils import construct_maze
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.envs.mujoco.maze.maze_env_utils import ray_segment_intersect, point_distance
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from rllab.misc import logger
from curriculum.envs.goal_env import GoalEnv, GoalExplorationEnv


class MazeEnv(ProxyEnv, Serializable):
    MODEL_CLASS = None
    ORI_IND = None

    MAZE_HEIGHT = None
    MAZE_SIZE_SCALING = None
    MAZE_MAKE_CONTACTS = False
    MAZE_STRUCTURE = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 'g', 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    MANUAL_COLLISION = False

    def __init__(
            self,
            # goal_generator,
            n_bins=20,
            sensor_range=10.,
            sensor_span=math.pi,
            maze_id=0,
            length=1,
            maze_height=0.5,
            maze_size_scaling=2,
            coef_inner_rew=0.,  # a coef of 0 gives no reward to the maze from the wrapped env.
            # goal_rew=1.,  # reward obtained when reaching the goal
            *args,
            **kwargs):
        Serializable.quick_init(self, locals())
        self._n_bins = n_bins
        self._sensor_range = sensor_range
        self._sensor_span = sensor_span
        self._maze_id = maze_id
        self.length = length
        self.coef_inner_rew = coef_inner_rew
        # self.goal_rew = goal_rew

        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self.MAZE_HEIGHT = height = maze_height
        self.MAZE_SIZE_SCALING = size_scaling = maze_size_scaling
        self.MAZE_STRUCTURE = structure = construct_maze(maze_id=self._maze_id, length=self.length)

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    # offset all coordinates so that robot starts at the origin
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (j * size_scaling - torso_x,
                                          i * size_scaling - torso_y,
                                          height / 2 * size_scaling),
                        size="%f %f %f" % (0.5 * size_scaling,
                                           0.5 * size_scaling,
                                           height / 2 * size_scaling),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 0.5"
                    )

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        if self.__class__.MAZE_MAKE_CONTACTS:
            contact = ET.SubElement(
                tree.find("."), "contact"
            )
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if str(structure[i][j]) == '1':
                        for geom in geoms:
                            ET.SubElement(
                                contact, "pair",
                                geom1=geom.attrib["name"],
                                geom2="block_%d_%d" % (i, j)
                            )

        _, file_path = tempfile.mkstemp(text=True)
        tree.write(file_path)  # here we write a temporal file with the robot specifications. Why not the original one??

        self._goal_range = self._find_goal_range()
        self._cached_segments = None

        inner_env = model_cls(file_path=file_path, *args, **kwargs)  # file to the robot specifications
        ProxyEnv.__init__(self, inner_env)  # here is where the robot env will be initialized
        # self.update_goal_generator(goal_generator)

    # @overrides
    # def update_goal_generator(self, goal_generator):
    #     self.wrapped_env.update_goal_generator(goal_generator)
    #     self._goal_generator = goal_generator

    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()

    def get_current_obs(self):
        return np.concatenate([self.wrapped_env.get_current_obs(),
                               ])

    @overrides  # TODO: do we need this?
    @property
    def goal_observation(self):
        return self.wrapped_env.goal_observation

    def get_ori(self):
        """
        First it tries to use a get_ori from the wrapped env. If not successfull, falls
        back to the default based on the ORI_IND specified in Maze (not accurate for quaternions)
        """
        try:
            return self.wrapped_env.wrapped_get_ori()
        except (NotImplementedError, AttributeError) as e:
            pass
        return self.wrapped_env.model.data.qpos[self.__class__.ORI_IND]

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    # space of only the robot observations (they go first in the get current obs) THIS COULD GO IN PROXYENV
    @property
    def robot_observation_space(self):
        shp = self.get_current_robot_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    # @property
    # def maze_observation_space(self):
    #     shp = self.get_current_maze_obs().shape
    #     ub = BIG * np.ones(shp)
    #     return spaces.Box(ub * -1, ub)

    def _find_robot(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling
        assert False

    def _find_goal_range(self):  # this only finds one goal!
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'g':
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    return minx, maxx, miny, maxy

    def _is_in_collision(self, pos):
        x, y = pos
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False

    def find_empty_space(self):
        structure = self.MAZE_STRUCTURE
        size_scaling = self.MAZE_SIZE_SCALING
        empty_space = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r' or structure[i][j] == 'g' or structure[i][j] == 0:
                    empty_space.append((j * size_scaling - self._init_torso_x, i * size_scaling - self._init_torso_y))
                    #return j * size_scaling, i * size_scaling
        return empty_space

    def is_feasible(self, pos):  # the arg is the goal, not the full space!!!
        empty_space = self.find_empty_space()
        for space in empty_space:
            if np.size(np.where(np.abs(np.array(np.array(pos).reshape(-1)[:2])-np.array(space)) < self.MAZE_SIZE_SCALING/2)[0]) == 2:
                # print("Pos {} is in empty space: {}".format(pos, space))
                return True
        return False

    @overrides
    def reset(self, *args, **kwargs):
        return self.wrapped_env.reset(*args, **kwargs)

    def step(self, action):
        if self.MANUAL_COLLISION:
            old_pos = self.wrapped_env.get_xy()
            inner_next_obs, inner_rew, done, info = self.wrapped_env.step(action)
            new_pos = self.wrapped_env.get_xy()
            if self._is_in_collision(new_pos):
                self.wrapped_env.set_xy(old_pos)
                done = False
        else:
            inner_next_obs, inner_rew, done, info = self.wrapped_env.step(action)
        next_obs = self.get_current_obs()

        reward = self.coef_inner_rew * inner_rew
        info['inner_rew'] = inner_rew

        return Step(next_obs, reward, done, **info)

    def action_from_key(self, key):
        return self.wrapped_env.action_from_key(key)

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        # we call here any logging related to the maze, strip the maze obs and call log_diag with the stripped paths
        # we need to log the purely maze reward!!
        # with logger.tabular_prefix('Maze_'):
        #     gather_undiscounted_returns = [sum(path['env_infos']['outer_rew']) for path in paths]
        #     logger.record_tabular_misc_stat('Return', gather_undiscounted_returns, placement='front')
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
            wrapped_undiscounted_return = np.mean([np.sum(path['env_infos']['inner_rew']) for path in paths])
            logger.record_tabular('AverageReturn', wrapped_undiscounted_return)
            self.wrapped_env.log_diagnostics(stripped_paths, *args, **kwargs)
