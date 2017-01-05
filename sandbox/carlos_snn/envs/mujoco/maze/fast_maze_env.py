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
from contextlib import contextmanager

from rllab import spaces
from rllab.envs.base import Step

from rllab.envs.proxy_env import ProxyEnv
# from sandbox.carlos_snn.envs.proxy_maze_env import ProxyMazeEnv

from rllab.core.serializable import Serializable
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.envs.mujoco.maze.maze_env_utils import ray_segment_intersect, point_distance
from rllab.envs.env_spec import EnvSpec
from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv

from rllab.misc.overrides import overrides
from rllab.misc import logger


class FastMazeEnv(MazeEnv, Serializable):
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
            *args,
            **kwargs):

        Serializable.quick_init(self, locals())
        MazeEnv.__init__(self, *args, **kwargs)
        self._blank_maze = False
        self.blank_maze_obs = np.concatenate([np.zeros(self._n_bins), np.zeros(self._n_bins)])

        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        self._observation_space = spaces.Box(ub * -1, ub)

        shp = self.get_current_robot_obs().shape
        ub = BIG * np.ones(shp)
        self._robot_observation_space = spaces.Box(ub * -1, ub)

        shp = self.get_current_maze_obs().shape
        ub = BIG * np.ones(shp)
        self._maze_observation_space = spaces.Box(ub * -1, ub)

    @overrides
    def get_current_maze_obs(self):
        # print('computing maze obs')
        # The observation would include both information about the robot itself as well as the sensors around its
        # environment
        structure = self.__class__.MAZE_STRUCTURE
        size_scaling = self.__class__.MAZE_SIZE_SCALING

        # compute origin cell i_o, j_o coordinates and center of it x_o, y_o (with 0,0 in the top-right corner of struc)
        o_xy = np.array(self._find_robot())  # this is self.init_torso_x, self.init_torso_y !!: center of the cell xy!
        o_ij = (o_xy / size_scaling).astype(int)  # this is the position in the grid (check if correct..)
        # print('the origin o_xy is: ', o_xy, ', and corresponds to grid o_ij: ', o_ij)

        robot_xy = np.array(self.wrapped_env.get_body_com("torso")[:2])  # the coordinates of this are wrt the init!!
        ori = self.get_ori()  # for Ant this is computed with atan2, which gives [-pi, pi]
        # print('the robot is in robot_xy: ', robot_xy, " with ori: ", ori)

        c_ij = o_ij + np.rint(robot_xy / size_scaling)
        c_xy = (c_ij - o_ij) * size_scaling  # the xy position of the current cell center in init_robot origin
        # print('the current cell c_xy is: ', c_xy, ', corresponding to c_ij: ', c_ij)

        R = int(self._sensor_range // size_scaling)

        wall_readings = np.zeros(self._n_bins)
        goal_readings = np.zeros(self._n_bins)

        for ray_idx in range(self._n_bins):
            ray_ori = ori - self._sensor_span * 0.5 + ray_idx / (
            self._n_bins - 1) * self._sensor_span  # make the ray in [-pi, pi]  #$%^&^%$#@
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
                next_i = int(c_ij[0] + x_dir * (r + 1))  # this is the i of the cells on the other side of the segment
                delta_y = (next_x - robot_xy[0]) * math.tan(ray_ori)
                y = robot_xy[1] + delta_y  # y of the intersection pt, wrt robot_init origin
                dist = np.sqrt(np.sum(np.square(robot_xy - (next_x, y))))
                # print('trying next_x: ', next_x, ', with next_i: ', next_i, ', yielding delta_y: ', delta_y,
                #       ', and hence y:', y, 'at a dist: ', dist)
                if dist <= self._sensor_range and 0 <= next_i < len(structure[0]):
                    j = int(np.rint((y + o_xy[1]) / size_scaling))
                    if 0 <= j < len(structure):
                        # print('the j is:', j)
                        if structure[j][next_i] == 1:
                            # print(next_i, j, ' is a wall\n')
                            wall_readings[ray_idx] = (self._sensor_range - dist) / self._sensor_range
                            # self.plot_ray(wall_readings[ray_idx], ray_idx)
                            break
                        elif structure[j][next_i] == 'g':  # remember to flip the ij when referring to the matrix!!
                            # print(j, next_i, ' is the GOAL\n')
                            goal_readings[ray_idx] = (self._sensor_range - dist) / self._sensor_range
                            # self.plot_ray(goal_readings[ray_idx], ray_idx, 'g')
                            break
                    else:
                        break
                else:
                    # print('too far\n')
                    break

            # same for next horizontal segment. If the distance is less (higher intensity), update the goal/wall reading
            for r in range(R):
                next_y = c_xy[1] + y_dir * (0.5 + r) * size_scaling  # y of the next horizontal segment
                next_j = int(
                    c_ij[1] + y_dir * (r + 1))  # this is the i and j of the cells on the other side of the segment
                # first check the intersection with the next horizontal segment:
                delta_x = (next_y - robot_xy[1]) / math.tan(ray_ori)
                x = robot_xy[0] + delta_x
                dist = np.sqrt(np.sum(np.square(robot_xy - (x, next_y))))
                # print('trying next_y: ', next_y, ', with next_j: ', next_j, ', yielding delta_x: ', delta_x,
                #       ', and hence x:', x, 'at a dist: ', dist)
                if dist <= self._sensor_range and 0 <= next_j < len(structure):
                    i = int(np.rint((x + o_xy[0]) / size_scaling))
                    if 0 <= i < len(structure[0]):
                        # print('the i is:', i)
                        intensity = (self._sensor_range - dist) / self._sensor_range
                        if structure[next_j][i] == 1:
                            if wall_readings[ray_idx] == 0 or intensity > wall_readings[ray_idx]:
                                wall_readings[ray_idx] = intensity
                                # self.plot_ray(wall_readings[ray_idx], ray_idx)
                            break
                        elif structure[next_j][i] == 'g':
                            # print(i, next_j, ' is the GOAL\n')
                            if goal_readings[ray_idx] == 0 or intensity > goal_readings[ray_idx]:
                                goal_readings[ray_idx] = intensity
                                # self.plot_ray(goal_readings[ray_idx], ray_idx, 'g')
                            break
                    else:
                        break
                else:
                    break

        # errase the goal readings behind a wall and the walls behind a goal:
        for n, wr in enumerate(wall_readings):
            if wr > goal_readings[n]:
                goal_readings[n] = 0
            elif wr <= goal_readings[n]:
                wall_readings[n] = 0

        obs = np.concatenate([
            wall_readings,
            goal_readings
        ])

        # random = np.random.randint(0, 10000)
        # if random == 1:
        #     self.plot_state('sensor plot for xy {}, {}'.format(*robot_xy))

        return obs

    @overrides
    def get_current_obs(self):
        if self._blank_maze:
            return np.concatenate([self.wrapped_env.get_current_obs(),
                                   self.blank_maze_obs
                                   ])
        else:
            # print('computing actual observation')
            return np.concatenate([self.wrapped_env.get_current_obs(),
                                   self.get_current_maze_obs()
                                   ])

    @contextmanager
    def blank_maze(self):
        previous_blank_maze_obs = self._blank_maze
        self._blank_maze = True
        yield
        self._blank_maze = previous_blank_maze_obs

    @property
    @overrides
    def observation_space(self):
        return self._observation_space

    # CF space of only the robot observations (they go first in the get current obs)
    @property
    @overrides
    def robot_observation_space(self):
        return self._robot_observation_space

    @property
    @overrides
    def maze_observation_space(self):
        return self._maze_observation_space

    def plot_ray(self, reading, ray_idx, color='r'):
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
        robot_xy = np.array(self.wrapped_env.get_body_com("torso")[:2])  # the coordinates of this are wrt the init!!
        ori = self.get_ori()  # for Ant this is computed with atan2, which gives [-pi, pi]

        ### I don't compute obs_maze here!

        # compute origin cell i_o, j_o coordinates and center of it x_o, y_o (with 0,0 in the top-right corner of struc)
        o_xy = np.array(self._find_robot())  # this is self.init_torso_x, self.init_torso_y !!: center of the cell xy!
        o_ij = (o_xy / size_scaling).astype(int)  # this is the position in the grid (check if correct..)

        o_xy_plot = o_xy / size_scaling * 2
        robot_xy_plot = o_xy_plot + robot_xy / size_scaling * 2

        plt.scatter(*robot_xy_plot)

        # for ray_idx in range(self._n_bins):
        length_wall = self._sensor_range - reading * self._sensor_range if reading else 1e-6
        ray_ori = ori - self._sensor_span * 0.5 + ray_idx / (self._n_bins - 1) * self._sensor_span
        if ray_ori > math.pi:
            ray_ori -= 2 * math.pi
        elif ray_ori < - math.pi:
            ray_ori += 2 * math.pi
        # find the end point wall
        end_xy = (robot_xy + length_wall * np.array([math.cos(ray_ori), math.sin(ray_ori)]))
        end_xy_plot = (o_ij + end_xy / size_scaling) * 2
        plt.plot([robot_xy_plot[0], end_xy_plot[0]], [robot_xy_plot[1], end_xy_plot[1]], color)

        ax.set_title('sensors debug')
        print('plotting now, close the window')
        # plt.show(fig)
        # plt.close()

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
