import os.path as osp
import tempfile
import xml.etree.ElementTree as ET
import math
from functools import reduce

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import patches
from matplotlib import pyplot as plt

import numpy as np
import collections

from rllab import spaces
from rllab.envs.base import Step

from rllab.envs.proxy_env import ProxyEnv
# from sandbox.carlos_snn.envs.proxy_maze_env import ProxyMazeEnv

from rllab.core.serializable import Serializable
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.envs.mujoco.maze.maze_env_utils import ray_segment_intersect, point_distance
from rllab.envs.env_spec import EnvSpec

from rllab.misc.overrides import overrides
from rllab.misc import logger


class MazeEnv(ProxyEnv, Serializable):
    MODEL_CLASS = None
    ORI_IND = None  # this is 3 for Ant

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
            n_bins=20,
            sensor_range=10.,
            sensor_span=math.pi,
            maze_id=0,
            length=1,
            maze_height=0.5,
            maze_size_scaling=2,
            coef_inner_rew=0.,  # a coef of 0 gives no reward to the maze from the wrapped env.
            goal_rew=1.,  # reward obtained when reaching the goal
            *args,
            **kwargs):

        Serializable.quick_init(self, locals())
        self._n_bins = n_bins
        self._sensor_range = sensor_range
        self._sensor_span = sensor_span
        self._maze_id = maze_id
        self.__class__.MAZE_HEIGHT = maze_height
        self.__class__.MAZE_SIZE_SCALING = maze_size_scaling
        self.coef_inner_rew = coef_inner_rew
        self.goal_rew = goal_rew

        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        size_scaling = self.__class__.MAZE_SIZE_SCALING
        height = self.__class__.MAZE_HEIGHT

        # define the maze to use
        if self._maze_id == 0:
            structure = self.__class__.MAZE_STRUCTURE
        elif self._maze_id == 1:  # donuts maze: can reach the single goal by 2 equal paths
            c = length + 4
            M = np.ones((c, c))
            M[1:c - 1, (1, c - 2)] = 0
            M[(1, c - 2), 1:c - 1] = 0
            M = M.astype(int).tolist()
            M[1][c // 2] = 'r'
            M[c - 2][c // 2] = 'g'
            structure = M
            print(self.__class__.MAZE_STRUCTURE)
            self.__class__.MAZE_STRUCTURE = structure
            print("the new one is", self.__class__.MAZE_STRUCTURE)

        elif self._maze_id == 2:  # spiral maze: need to use all the keys (only makes sense for length >=3)
            c = length + 4
            M = np.ones((c, c))
            M[1:c - 1, (1, c - 2)] = 0
            M[(1, c - 2), 1:c - 1] = 0
            M = M.astype(int).tolist()
            M[1][c // 2] = 'r'
            # now block one of the ways and put the goal on the other side
            M[1][c // 2 - 1] = 1
            M[1][c // 2 - 2] = 'g'
            structure = M
            self.__class__.MAZE_STRUCTURE = structure
            print(structure)

        elif self._maze_id == 3:  # corridor with goals at the 2 extremes
            structure = [
                [1] * (2 * length + 5),
                [1, 'g'] + [0] * length + ['r'] + [0] * length + ['g', 1],
                [1] * (2 * length + 5),
            ]
            self.__class__.MAZE_STRUCTURE = structure
            print(structure)

        elif 4 <= self._maze_id <= 7:  # cross corridor, goal in
            c = 2 * length + 5
            M = np.ones((c, c))
            M = M - np.diag(np.ones(c))
            M = M - np.diag(np.ones(c - 1), 1) - np.diag(np.ones(c - 1), -1)
            i = np.arange(c)
            j = i[::-1]
            M[i, j] = 0
            M[i[:-1], j[1:]] = 0
            M[i[1:], j[:-1]] = 0
            M[np.array([0, c - 1]), :] = 1
            M[:, np.array([0, c - 1])] = 1
            M = M.astype(int).tolist()
            M[c // 2][c // 2] = 'r'
            # for i in [1, c - 2]:
            #     for j in [1, c - 2]:
            #         M[i][j] = 'g'
            if self._maze_id == 4:
                M[1][1] = 'g'
            if self._maze_id == 5:
                M[1][c - 2] = 'g'
            if self._maze_id == 6:
                M[c - 2][1] = 'g'
            if self._maze_id == 7:
                M[c - 2][c - 2] = 'g'
            structure = M
            self.__class__.MAZE_STRUCTURE = structure
            print(structure)

        elif self._maze_id == 8:  # reflexion of benchmark maze
            structure = [
                [1, 1, 1, 1, 1],
                [1, 'g', 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 'r', 0, 0, 1],
                [1, 1, 1, 1, 1],
            ]
            self.__class__.MAZE_STRUCTURE = structure
            print(structure)

        elif self._maze_id == 9:  # sym benchmark maze
            structure = [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 'r', 1],
                [1, 0, 1, 1, 1],
                [1, 0, 0, 'g', 1],
                [1, 1, 1, 1, 1],
            ]
            self.__class__.MAZE_STRUCTURE = structure
            print(structure)

        elif self._maze_id == 10:  # reflexion of sym of benchmark maze
            structure = [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 'g', 1],
                [1, 0, 1, 1, 1],
                [1, 0, 0, 'r', 1],
                [1, 1, 1, 1, 1],
            ]
            self.__class__.MAZE_STRUCTURE = structure
            print(structure)

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
                        rgba="0.4 0.4 0.4 1"
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

        inner_env = model_cls(*args, file_path=file_path, **kwargs)  # file to the robot specifications
        ProxyEnv.__init__(self, inner_env)  # here is where the robot env will be initialized

    def reset(self):
        self.wrapped_env.reset()
        return self.get_current_obs()

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
        x, y = self.wrapped_env.get_body_com("torso")[:2]
        # ref_x = x + self._init_torso_x
        # ref_y = y + self._init_torso_y
        info['maze_rewards'] = 0
        info['inner_rew'] = inner_rew
        reward = self.coef_inner_rew * inner_rew
        minx, maxx, miny, maxy = self._goal_range
        # print("goal range: x [%s,%s], y [%s,%s], now [%s,%s]" % (str(minx), str(maxx), str(miny), str(maxy),
        #                                                          str(x), str(y)))
        if minx <= x <= maxx and miny <= y <= maxy:
            done = True
            reward += self.goal_rew
            info[
                'maze_rewards'] = 1  # we keep here the original one, so that the AvgReturn is directly the freq of success
        return Step(next_obs, reward, done, **info)

    def get_current_maze_obs(self):
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING

        # compute origin cell i_o, j_o coordinates and center of it x_o, y_o (with 0,0 in the top-right corner of struc)
        o_xy = np.array(self._find_robot())  # this is self.init_torso_x, self.init_torso_y !!: center of the cell xy!
        o_ij = (o_xy / size_scaling).astype(int)  # this is the position in the grid (check if correct..)
        # print('the origin xy is: ', o_xy, ', and corresponds to grid: ', o_ij)

        robot_xy = np.array(self.wrapped_env.get_body_com("torso")[:2])  # the coordinates of this are wrt the init!!
        ori = self.get_ori()  # for Ant this is computed with atan2, which gives [-pi, pi]
        # print('the robot is in: ', robot_xy, " with ori: ", ori)

        c_ij = o_ij + np.rint(robot_xy / size_scaling)
        c_xy = (c_ij - o_ij) * size_scaling  # the xy position of the current cell center in init_robot origin
        # print('the current cell xy is: ', c_xy, ', corresponding to ij: ', c_ij)

        R = int(self._sensor_range // size_scaling)

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + ray_idx / (self._n_bins - 1) * self._sensor_span  # make the ray in [-pi, pi]  #$%^&^%$#@
            if ray_ori > math.pi:
                ray_ori -= 2 * math.pi
            elif ray_ori < - math.pi:
                ray_ori += 2 * math.pi
            x_dir, y_dir = 1, 1
            if math.pi / 2. <= ray_ori <= math.pi:
                x_dir = -1
            elif 0 > ray_ori >= - math.pi / 2.:
                y_dir = -1
            elif - math.pi / 2. > ray_ori >= - math.pi:
                x_dir, y_dir = -1, -1
            # print('the ray_ori is: ', ray_ori, ", corresponding to dir:", (x_dir, y_dir))

            for r in range(R):
                next_x = c_xy[0] + x_dir * (0.5 + r) * size_scaling  # x of the next vertical segment, in init_rob coord
                next_i = int(c_ij[0] + r + x_dir)  # this is the i of the cells on the other side of the segment
                delta_y = (next_x - robot_xy[0]) * math.tan(ray_ori)
                y = robot_xy[1] + delta_y  # y of the intersection pt, wrt robot_init origin
                dist = np.sqrt(np.sum(np.square(robot_xy - (next_x, y))))
                # print('trying next_x: ', next_x, ', with next_i: ', next_i, ', yielding delta_y: ', delta_y,
                #       ', and hence y:', y, 'at a dist: ', dist)
                if dist <= self._sensor_range and 0 <= next_i < len(structure[0]):
                    j = int((y + o_xy[1] + 0.5 * size_scaling) / size_scaling)
                    if 0 <= j < len(structure):
                        # print('the j is:', j)
                        if structure[j][next_i] == 1:
                            # print(next_i, j, ' is a wall\n')
                            wall_readings[ray_idx] = (self._sensor_range - dist) / self._sensor_range
                            break
                        elif structure[j][next_i] == 'g':  # remember to flip the ij when referring to the matrix!!
                            # print(j, next_i, ' is the GOAL\n')
                            goal_readings[ray_idx] = (self._sensor_range - dist) / self._sensor_range
                            break
                    else:
                        break
                else:
                    # print('too far\n')
                    break

            # same for next horizontal segment. If the distance is less (higher intensity), update the goal/wall reading
            for r in range(R):
                next_y = c_xy[1] + y_dir * (0.5 + r) * size_scaling  # y of the next horizontal segment
                next_j = int(c_ij[1] + r + y_dir)  # this is the i and j of the cells on the other side of the segment
                # first check the intersection with the next horizontal segment:
                delta_x = (next_y - robot_xy[1]) / math.tan(ray_ori)
                x = robot_xy[0] + delta_x
                dist = np.sqrt(np.sum(np.square(robot_xy - (x, next_y))))
                if dist <= self._sensor_range and 0 <= next_j < len(structure):
                    i = int((x + o_xy[0] + 0.5 * size_scaling) / size_scaling)
                    if 0 <= i < len(structure[0]):
                        intensity = (self._sensor_range - dist) / self._sensor_range
                        if structure[next_j][i] == 1:
                            if wall_readings[ray_idx] == 0 or intensity > wall_readings[ray_idx]:
                                wall_readings[ray_idx] = intensity
                            break
                        elif structure[next_j][i] == 'g':
                            if goal_readings[ray_idx] == 0 or intensity > goal_readings[ray_idx]:
                                goal_readings[ray_idx] = intensity
                            break
                    else:
                        break
                else:
                    break

        # errase the goal readings behind a wall:
        for n, wr in enumerate(wall_readings):
            if wr > 0 and wr > goal_readings[n]:  # if the goal behind a wall, errase. But not if goal in front of a wall
                goal_readings[n] = 0

        obs = np.concatenate([
            wall_readings,
            goal_readings
        ])

        random = np.random.randint(0, 1000)
        if random == 1:
            self.plot_state('sensor plot for xy {}, {}'.format(*robot_xy))

        return obs

    def find_y(self, robot_xy, ray_ori, next_x):
        return robot_xy[1] + (next_x - robot_xy[0]) * math.tan(ray_ori)

    def find_x(self, robot_xy, ray_ori, next_y):
        return robot_xy[0] + (next_y - robot_xy[1]) / math.tan(ray_ori)

    def get_current_robot_obs(self):
        return self.wrapped_env.get_current_obs()

    def get_current_obs(self):
        return np.concatenate([self.wrapped_env.get_current_obs(),
                               self.get_current_maze_obs()
                               ])

    def get_ori(self):
        return self.wrapped_env.model.data.qpos[self.__class__.ORI_IND]

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
        shp = self.get_current_maze_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    @overrides
    def spec(self):
        return EnvSpec(
            observation_space=self.observation_space,
            # maze_observation_space=self.maze_observation_space,
            # robot_observation_space=self.robot_observation_space,
            action_space=self.action_space,
        )

    def action_from_key(self, key):
        return self.wrapped_env.action_from_key(key)

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    def _find_robot(self):
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 'r':
                    return j * size_scaling, i * size_scaling
        assert False

    def _find_goal_range(self):  # this only finds one goal!  xy coord of all bounding seg., with origin in robot init
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
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
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
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

    @overrides
    def log_diagnostics(self, paths, *args, **kwargs):
        # we call here any logging related to the maze, strip the maze obs and call log_diag with the stripped paths
        # we need to log the purely gather reward!!
        with logger.tabular_prefix('Maze_'):
            gather_undiscounted_returns = [sum(path['env_infos']['maze_rewards']) for path in paths]
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

    def plot_state(self, name='sensors', state=None):
        if state:
            self.wrapped_env.reset(state)

        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING
        # duplicate cells to plot the maze
        structure_plot = np.zeros(((len(structure) - 1) * 2, (len(structure[0]) - 1) * 2))
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                cell = structure[i][j]
                if type(cell) is not int:
                    cell = 0.3 if cell == 'r' else 0.7
                if i == 0:
                    if j == 0:
                        structure_plot[i, j] = cell
                    elif j == len(structure[0]) - 1:
                        structure_plot[i, 2 * j - 1] = cell
                    else:
                        structure_plot[i, 2 * j - 1:2 * j + 1] = cell
                elif i == len(structure) - 1:
                    if j == 0:
                        structure_plot[2 * i - 1, j] = cell
                    elif j == len(structure[0]) - 1:
                        structure_plot[2 * i - 1, 2 * j - 1] = cell
                    else:
                        structure_plot[2 * i - 1, 2 * j - 1:2 * j + 1] = cell
                else:
                    if j == 0:
                        structure_plot[2 * i - 1:2 * i + 1, j] = cell
                    elif j == len(structure[0]) - 1:
                        structure_plot[2 * i - 1:2 * i + 1, 2 * j - 1] = cell
                    else:
                        structure_plot[2 * i - 1:2 * i + 1, 2 * j - 1:2 * j + 1] = cell

        fig, ax = plt.subplots()
        im = ax.pcolor(-np.array(structure_plot), cmap='gray', edgecolor='black', linestyle=':', lw=1)
        x_labels = list(range(len(structure[0])))
        y_labels = list(range(len(structure)))
        ax.grid(True)  # elimiate this to avoid inner lines

        ax.xaxis.set(ticks=2 * np.arange(len(x_labels)), ticklabels=x_labels)
        ax.yaxis.set(ticks=2 * np.arange(len(y_labels)), ticklabels=y_labels)
        ########
        obs = self.get_current_maze_obs()

        robot_xy = np.array(self.wrapped_env.get_body_com("torso")[:2])  # the coordinates of this are wrt the init!!
        ori = self.get_ori()  # for Ant this is computed with atan2, which gives [-pi, pi]

        # compute origin cell i_o, j_o coordinates and center of it x_o, y_o (with 0,0 in the top-right corner of struc)
        o_xy = np.array(self._find_robot())  # this is self.init_torso_x, self.init_torso_y !!: center of the cell xy!
        o_ij = (o_xy / size_scaling).astype(int)  # this is the position in the grid (check if correct..)

        o_xy_plot = o_xy / size_scaling * 2
        robot_xy_plot = o_xy_plot + robot_xy / size_scaling * 2

        plt.scatter(*robot_xy_plot)

        for ray_idx in range(self._n_bins):
            length_wall = self._sensor_range - obs[ray_idx] * self._sensor_range if obs[ray_idx] else 1e-6
            ray_ori = ori - self._sensor_span * 0.5 + ray_idx / (self._n_bins - 1) * self._sensor_span
            if ray_ori > math.pi:
                ray_ori -= 2 * math.pi
            elif ray_ori < - math.pi:
                ray_ori += 2 * math.pi
            # find the end point wall
            end_xy = (robot_xy + length_wall * np.array([math.cos(ray_ori), math.sin(ray_ori)]))
            end_xy_plot = (o_ij + end_xy / size_scaling) * 2
            plt.plot([robot_xy_plot[0], end_xy_plot[0]], [robot_xy_plot[1], end_xy_plot[1]], 'r')

            length_goal = self._sensor_range - obs[ray_idx + self._n_bins] * self._sensor_range if obs[
                ray_idx + self._n_bins] else 1e-6
            ray_ori = ori - self._sensor_span * 0.5 + ray_idx / (self._n_bins - 1) * self._sensor_span
            # find the end point goal
            end_xy = (robot_xy + length_goal * np.array([math.cos(ray_ori), math.sin(ray_ori)]))
            end_xy_plot = (o_ij + end_xy / size_scaling) * 2
            plt.plot([robot_xy_plot[0], end_xy_plot[0]], [robot_xy_plot[1], end_xy_plot[1]], 'g')

        log_dir = logger.get_snapshot_dir()
        ax.set_title('sensors: ' + name)

        plt.savefig(osp.join(log_dir, name + '_sesors.png'))  # this saves the current figure, here f
        plt.close()
