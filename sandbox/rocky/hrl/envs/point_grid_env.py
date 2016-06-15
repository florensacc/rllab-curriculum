from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.grid_world_env import GridWorldEnv
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.spaces.product import Product
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
import numpy as np
import math

UP_ACTION = GridWorldEnv.action_from_direction("up")
DOWN_ACTION = GridWorldEnv.action_from_direction("down")
LEFT_ACTION = GridWorldEnv.action_from_direction("left")
RIGHT_ACTION = GridWorldEnv.action_from_direction("right")


def line_intersect(pt1, pt2, ptA, ptB, tol=1e-8):
    """
    Taken from https://www.cs.hmc.edu/ACM/lectures/intersections.html

    this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

    returns a tuple: (xi, yi, valid, r, s), where
    (xi, yi) is the intersection
    r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
    s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
    valid == 0 if there are 0 or inf. intersections (invalid)
    valid == 1 if it has a unique intersection ON the segment
    """

    DET_TOLERANCE = tol

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def safe_move(pos, inc, block_size, tol=1e-8):
    segments = [
        # top left corner
        [(0, block_size), (block_size, block_size)],
        [(block_size, 0), (block_size, block_size)],

        # bottom left corner
        [(1 - block_size, block_size), (1, block_size)],
        [(1 - block_size, 0), (1 - block_size, block_size)],

        # top right corner
        [(0, 1 - block_size), (block_size, 1 - block_size)],
        [(block_size, 1 - block_size), (block_size, 1)],

        # bottom right corner
        [(1 - block_size, 1 - block_size), (1 - block_size, 1)],
        [(1 - block_size, 1 - block_size), (1, 1 - block_size)],
    ]

    new_loc = pos + inc

    pos_x, pos_y = pos

    # move_segment = [tuple(pos), tuple(new_loc)]

    closest_pt = None
    closest_dist_sqr = None

    for to_A, to_B in segments:
        xi, yi, par, r, s = line_intersect(pos, new_loc, to_A, to_B, tol)
        if par == 0:
            pass
        elif tol <= r <= 1 + tol and -tol <= s <= 1 + tol:
            # intersect! (but ignore intersection at the origin point)
            dist_sqr = (pos_x - xi) ** 2 + (pos_y - yi) ** 2
            if closest_dist_sqr is None or dist_sqr < closest_dist_sqr:
                closest_dist_sqr = dist_sqr
                closest_pt = (xi, yi)
        else:
            pass

    if closest_pt is not None:
        return closest_pt
    # still need to check possible intersections at the origin. we only need to check if new_loc is in the block region
    newx, newy = new_loc
    if (newx <= block_size and newy <= block_size) or \
            (newx <= block_size and 1 - block_size <= newy) or \
            (1 - block_size <= newx and newy <= block_size) or \
            (1 - block_size <= newx and 1 - block_size <= newy):
        return pos
    return new_loc


class PointGridEnv(GridWorldEnv, Serializable):
    """
    This environment wraps a discrete grid world environment into a continuous one, where the discrete navigation
    agent is replaced by a point robot, controllable by specifying its velocity.
    """

    def __init__(self, desc='4x4', speed=0.3, obs_type="hybrid", block_size=0.25):
        """
        :param desc:
        :param speed:
        :param obs_type:
            - hybrid: have both discrete position (one-hot representation), and continuous position relative to the
            discrete grid
            - cont_only: have only a continuous position in the global coordinate system
        :param block_size: to prevent diagonal crossing, we place blocks at the corners of each grid. This parameter
        specifies the size of the block.
        :return:
        """
        Serializable.quick_init(self, locals())
        GridWorldEnv.__init__(self, desc)
        # stores the relative coordinate w.r.t. the current grid
        self.grid_coords = None
        self.obs_type = obs_type
        if obs_type == "hybrid":
            self._observation_space = Product(
                Discrete(self.n_row * self.n_col),
                Box(low=-0.01, high=1.01, shape=(2,))
            )
        else:
            raise NotImplementedError
        self._action_space = Box(low=-1., high=1., shape=(2,))
        self.speed = speed
        self.block_size = block_size
        self.viewer = None

    def reset(self):
        GridWorldEnv.reset(self)
        self.grid_coords = np.array([0.5, 0.5])
        return self.get_current_obs()

    def get_current_obs(self):
        if self.obs_type == "hybrid":
            return self.state, np.copy(self.grid_coords)
        else:
            raise NotImplementedError

    def step(self, action):
        scaled_action = np.clip(np.array(action), -1, 1) * self.speed
        cx, cy = safe_move(self.grid_coords, scaled_action, self.block_size)
        # update the state
        prev_state = self.state
        reward = 0
        done = False

        moves = []
        if cx < 0:
            moves.append(UP_ACTION)
        elif cx > 1:
            moves.append(DOWN_ACTION)
        if cy < 0:
            moves.append(LEFT_ACTION)
        elif cy > 1:
            moves.append(RIGHT_ACTION)
        assert len(moves) <= 1
        if len(moves) > 0:
            move = moves[0]
            _, reward, done, _ = GridWorldEnv.step(self, action=move)
            if self.state == prev_state:
                # if not moved, either we hit a wall or the border of the grid world
                # clip the position
                cx = np.clip(cx, 0., 1.)
                cy = np.clip(cy, 0., 1.)
            else:
                # if moved, update the position so that it lies in [0, 1] again
                if cx < 0:
                    cx += 1
                elif cx > 1:
                    cx -= 1
                if cy < 0:
                    cy += 1
                elif cy > 1:
                    cy -= 1
        self.grid_coords = np.array([cx, cy])
        return Step(self.get_current_obs(), reward, done)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def render(self):
        from .gym_renderer import Viewer
        import pyglet

        if self.viewer is None:
            self.viewer = Viewer(500, 500)
            self.viewer.set_bounds(-1, self.n_col + 1, -1, self.n_row + 1)
        for row_idx in range(self.n_row + 1):
            self.viewer.draw_line((0., row_idx), (self.n_col, row_idx))
        for col_idx in range(self.n_col + 1):
            self.viewer.draw_line((col_idx, 0.), (col_idx, self.n_row))
        for row_idx in range(self.n_row):
            for col_idx in range(self.n_col):
                entry = self.desc[row_idx][col_idx]
                if row_idx * self.n_col + col_idx == self.state:
                    cx, cy = self.grid_coords
                    self.viewer.draw_circle(
                        radius=0.05,
                        center=(col_idx + cy, self.n_row - row_idx - cx),
                        color=(0, 0, 255)
                    )
                if entry == 'G':
                    self.viewer.draw_polygon(
                        v=[
                            (col_idx, self.n_row - row_idx),
                            (col_idx, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx),
                        ],
                        color=(0, 255, 0)
                    )
                    # self.viewer.draw_circle(
                    #     radius=0.05,
                    #     center=(col_idx + 0.5, self.n_row - row_idx - 0.5),
                    #     color=(0, 255, 0)
                    # )
                elif entry == 'H':
                    self.viewer.draw_polygon(
                        v=[
                            (col_idx, self.n_row - row_idx),
                            (col_idx, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx),
                        ],
                        color=(255, 0, 0)
                    )
                    continue
                self.viewer.draw_polygon(
                    v=[
                        (col_idx, self.n_row - row_idx),
                        (col_idx, self.n_row - row_idx - self.block_size),
                        (col_idx + self.block_size, self.n_row - row_idx - self.block_size),
                        (col_idx + self.block_size, self.n_row - row_idx),
                    ],
                    color=(0, 0, 0)
                )
                self.viewer.draw_polygon(
                    v=[
                        (col_idx + 1 - self.block_size, self.n_row - row_idx),
                        (col_idx + 1 - self.block_size, self.n_row - row_idx - self.block_size),
                        (col_idx + 1, self.n_row - row_idx - self.block_size),
                        (col_idx + 1, self.n_row - row_idx),
                    ],
                    color=(0, 0, 0)
                )
                self.viewer.draw_polygon(
                    v=[
                        (col_idx + 1 - self.block_size, self.n_row - row_idx - 1 + self.block_size),
                        (col_idx + 1 - self.block_size, self.n_row - row_idx - 1),
                        (col_idx + 1, self.n_row - row_idx - 1),
                        (col_idx + 1, self.n_row - row_idx - 1 + self.block_size),
                    ],
                    color=(0, 0, 0)
                )
                self.viewer.draw_polygon(
                    v=[
                        (col_idx, self.n_row - row_idx - 1 + self.block_size),
                        (col_idx, self.n_row - row_idx - 1),
                        (col_idx + self.block_size, self.n_row - row_idx - 1),
                        (col_idx + self.block_size, self.n_row - row_idx - 1 + self.block_size),
                    ],
                    color=(0, 0, 0)
                )
        self.viewer.render()
        self.viewer.window.dispatch_events()
        self.viewer.window.flip()

    def start_interactive(self):
        import pyglet
        # if self.viewer is None:
        self.render()

        key = pyglet.window.key
        keys = pyglet.window.key.KeyStateHandler()
        self.viewer.window.push_handlers(keys)

        @self.viewer.window.event
        def on_draw():

        # @self.viewer.window.event
        # def on_key_press(key, modifiers):
            if keys[key.UP]:
                action = [-1, 0]
            elif keys[key.DOWN]:
                action = [1, 0]
            elif keys[key.LEFT]:
                action = [0, -1]
            elif keys[key.RIGHT]:
                action = [0, 1]
            else:
                action = None
            if action is not None:
                _, _, done, _ = self.step(action)
                if done:
                    print("resetting")
                    self.reset()
            self.render()
            # print(key)
            # pass

        pyglet.app.run()
