import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import math

from rllab.misc import logger

def construct_maze(maze_id=0, length=1):

    # define the maze to use
    if maze_id == 0:
        if length != 1:
            raise NotImplementedError("Maze_id 0 only has length 1!")
        structure = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 'g', 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    elif maze_id == 1:  # donuts maze: can reach the single goal by 2 equal paths
        c = length + 4
        M = np.ones((c, c))
        M[1:c - 1, (1, c - 2)] = 0
        M[(1, c - 2), 1:c - 1] = 0
        M = M.astype(int).tolist()
        M[1][c // 2] = 'r'
        M[c - 2][c // 2] = 'g'
        structure = M

    elif maze_id == 2:  # spiral maze: need to use all the keys (only makes sense for length >=3)
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

    elif maze_id == 3:  # corridor with goals at the 2 extremes
        structure = [
            [1] * (2 * length + 5),
            [1, 'g'] + [0] * length + ['r'] + [0] * length + ['g', 1],
            [1] * (2 * length + 5),
            ]

    elif 4 <= maze_id <= 7:  # cross corridor, goal in
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
        if maze_id == 4:
            M[1][1] = 'g'
        if maze_id == 5:
            M[1][c - 2] = 'g'
        if maze_id == 6:
            M[c - 2][1] = 'g'
        if maze_id == 7:
            M[c - 2][c - 2] = 'g'
        structure = M

    elif maze_id == 8:  # reflexion of benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 'g', 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 'r', 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

    elif maze_id == 9:  # sym benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 'r', 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 'g', 1],
            [1, 1, 1, 1, 1],
        ]

    elif maze_id == 10:  # reflexion of sym of benchmark maze
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 'g', 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 'r', 1],
            [1, 1, 1, 1, 1],
        ]
    if structure:
        return structure
    else:
        raise NotImplementedError("The provided MazeId is not recognized")


def plot_ray(self, reading, ray_idx, color='r'):
    structure = self.MAZE_STRUCTURE
    size_scaling = self.MAZE_SIZE_SCALING
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
