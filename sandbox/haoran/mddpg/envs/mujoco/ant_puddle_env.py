from sandbox.tuomas.mddpg.envs.mujoco.mujoco_env import MujocoEnv
from sandbox.haoran.myscripts.quaternion import Quaternion
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from rllab.core.serializable import Serializable
from rllab.mujoco_py import MjViewer
import numpy as np
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

class Puddle(Serializable):
    """
    A puddle is a rectangular high-cost region
    """
    def __init__(self, x, y, width, height, angle, plot_args, cost, hard=False,
        text="", depth=2):
        """
        x, y: coordinate of the lower left corner
        """
        Serializable.quick_init(self, locals())
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        if angle != 0.:
            raise NotImplementedError
        self.angle = angle
        assert "color" in plot_args
        self.plot_args = plot_args
        self.cost = cost
        self.hard = hard
        self.text = text
        self.depth = depth

    def plot(self, ax):
        ax.add_patch(Rectangle(
            xy=(self.x, self.y),
            width=self.width,
            height=self.height,
            angle=self.angle,
            **self.plot_args
        ))
        ax.text(
            self.x + self.width/2,
            self.y + self.height/2,
            self.text,
        )

    def is_inside(self, pos):
        x, y = pos
        return all([
            self.x <= x,
            self.x + self.width >= x,
            self.y <= y,
            self.y + self.height >= y,
        ])

    def compute_cost(self, pos):
        if self.is_inside(pos):
            cost = self.cost
        else:
            cost = 0.
        return cost

class AntPuddleEnv(MujocoEnv, Serializable):

    def __init__(self,
            puddles=[],
            reward_type="goal",
            direction=None,
            goal=(10., 0.),
            flip_thr=0.,
            init_reward=1.,
            speed_coeff=1.,
            mujoco_env_args=dict(),
            plot_settings=None,
            goal_reward=1000,
        ):
        if direction is not None:
            assert np.isclose(np.linalg.norm(direction), 1.)
        self.puddles = puddles
        self.reward_type = reward_type
        self.direction = direction
        self.goal = goal
        self.init_reward = init_reward
        self.speed_coeff = speed_coeff
        self.flip_thr = flip_thr
        self.plot_settings = plot_settings
        self.goal_reward = goal_reward

        # dynamically generate and load the xml file
        self.file_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "ant_puddle_temporary.xml",
        )
        self.generate_xml_file()
        mujoco_env_args["file_path"] = self.file_path
        super().__init__(**mujoco_env_args)
        Serializable.quick_init(self, locals())

    def get_param_values(self):
        params = dict(
            puddles=self.puddles,
            direction=self.direction,
            reward_type=self.reward_type,
            flip_thr=self.flip_thr,
            goal=self.goal,
            speed_coeff=self.speed_coeff,
        )
        return params

    def set_param_values(self, params):
        self.__dict__.update(params)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")[:2]
        com = self.get_body_com("torso")[:2]
        if self.reward_type == "velocity":
            if self.direction is not None:
                motion_reward = comvel.dot(np.array(self.direction))
            else:
                motion_reward = np.linalg.norm(comvel)
        elif self.reward_type == "distance_from_origin":
            motion_reward = np.linalg.norm(com)
        elif self.reward_type == "goal":
            # reward y = a * exp(-b * x^2), x is distance
            x0 = np.linalg.norm(np.array(self.goal)) # init dist
            y0 = self.init_reward
            y1 = self.goal_reward

            a = y1
            b = -1. / (x0 ** 2) * np.log(y0 / y1)
            x = np.linalg.norm(com - np.array(self.goal)) # dist to goal
            motion_reward = a * np.exp(-b * (x ** 2))

            # print(a, b)
            # print(x, motion_reward)
            motion_reward += self.speed_coeff * np.linalg.norm(comvel)
        else:
            raise NotImplementedError

        state = self._state

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5

        action_violations = np.maximum(np.maximum(lb - action, action - ub), 0)
        action_violation_cost = np.sum((action_violations / scaling)**2)

        ctrl_cost = 0.5 * 1e-3 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-4 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 0.05
        pos = state[:2] # 2D position
        puddle_cost = self.compute_puddle_cost(pos)
        reward = (motion_reward - ctrl_cost - contact_cost + survive_reward
                  - action_violation_cost - puddle_cost)


        # q describes the orientation of the ball
        q = Quaternion(*tuple(self.model.data.qpos[3:7].ravel()))
        # z is the z-pos of the bottom of the ball after applying q
        z = q.rotate(np.array([0., 0., -1.]))[2]

        notdone = all([
            np.isfinite(state).all(),
            state[2] <= 1.0, # prevent jumpping
            z < self.flip_thr, # prevent flipping
        ])
        done = not notdone

        ob = self.get_current_obs()

        return Step(ob, float(reward), done, com=self.get_body_com("torso"))

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(
            'env: ForwardProgressAverage', np.mean(progs))
        logger.record_tabular(
            'env: ForwardProgressMax', np.max(progs))
        logger.record_tabular(
            'env: ForwardProgressMin', np.min(progs))
        logger.record_tabular(
            'env: ForwardProgressStd', np.std(progs))

    def log_stats(self, algo, epoch, paths):
        # forward distance
        progs = []
        for path in paths:
            coms = path["env_infos"]["com"]
            progs.append(coms[-1][0] - coms[0][0])
                # x-coord of com at the last time step minus the 1st step
        stats = {
            'env: ForwardProgressAverage': np.mean(progs),
            'env: ForwardProgressMax': np.max(progs),
            'env: ForwardProgressMin': np.min(progs),
            'env: ForwardProgressStd': np.std(progs),
        }

        if self.reward_type == "distance_from_origin":
            dists = []
            for path in paths:
                pos = path["env_infos"]["com"][-1][:2] # (x,y) of last time step
                dists.append(np.linalg.norm(pos))
            stats = {
                'env: FinalDistanceFromOriginAverage': np.mean(dists),
                'env: FinalDistanceFromOriginMax': np.max(dists),
                'env: FinalDistanceFromOriginMin': np.min(dists),
                'env: FinalDistanceFromOriginStd': np.std(dists),
            }
        elif self.reward_type == "goal":
            dists = [
                np.linalg.norm(
                    path["env_infos"]["com"][-1][:2] - np.array(self.goal)
                )
                for path in paths
            ]
            stats = {
                'env: FinalDistanceFromGoalAverage': np.mean(dists),
                'env: FinalDistanceFromGoalMax': np.max(dists),
                'env: FinalDistanceFromGoalMin': np.min(dists),
                'env: FinalDistanceFromGoalStd': np.std(dists),
            }
        elif self.reward_type == "velocity":
            pass
        else:
            raise NotImplementedError

        return stats

    def compute_puddle_cost(self, pos):
        cost = 0
        for puddle in self.puddles:
            cost += puddle.compute_cost(pos)
        return cost

    def plot_puddles(self, ax):
        for puddle in self.puddles:
            puddle.plot(ax)

    def plot_goal(self, ax):
        x, y = self.goal
        ax.plot(x, y, 'b*', markersize=10)
        if self.reward_type == "goal":
            xmin, xmax = self.plot_settings["xlim"]
            ymin, ymax = self.plot_settings["ylim"]
            xx = np.arange(xmin, xmax, 0.1)
            yy = np.arange(ymin, ymax, 0.1)
            X, Y = np.meshgrid(xx, yy)
            goal_x, goal_y = self.goal
            D_square = (X - goal_x) ** 2 + (Y - goal_y) ** 2 # dist square

            x0 = np.linalg.norm(np.array(self.goal)) # init dist
            y0 = self.init_reward
            y1 = self.goal_reward
            a = y1
            b = -1. / (x0 ** 2) * np.log(y0 / y1)
            R = a * np.exp(-b * (D_square))

            cs = ax.contour(X, Y, R, 20)
            ax.clabel(cs, inline=1, fontsize=10, fmt='%.1f')
        plt.axis("equal")


    def plot_env(self, ax):
        self.plot_puddles(ax)
        if self.reward_type == "goal":
            self.plot_goal(ax)
        ax.plot(0, 0, 'go')

        if self.plot_settings is not None:
            xlim = self.plot_settings["xlim"]
            ylim = self.plot_settings["ylim"]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True)
            ax.set_xticks(np.arange(np.floor(xlim[0]), np.ceil(xlim[1])))
            ax.set_yticks(np.arange(np.floor(ylim[0]), np.ceil(ylim[1])))

    @overrides
    def plot_paths(self, paths, ax):
        ax.grid(True)
        self.plot_env(ax)
        for path in paths:
            positions = path["env_infos"]["com"]
            xx = positions[:, 0]
            yy = positions[:, 1]
            ax.plot(xx, yy, 'b')

    def generate_xml_file(self):
        # avoid re-generating the xml file during parallel sampling
        import multiprocessing as mp
        if mp.current_process().name != "MainProcess":
            return

        # read the template
        from lxml.etree import Element as ET
        from lxml import etree

        self.template_file_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            "ant_puddle_template.xml",
        )
        with open(self.template_file_path) as f:
            template_string = f.read()
        root = etree.fromstring(template_string)
        worldbody = root[5]

        # add the puddles
        for puddle in self.puddles:
            r, g, b, a = puddle.plot_args["color"]

            if puddle.hard:
                worldbody.append(ET('geom',
                    type="box",
                    pos="{center_x} {center_y} {center_z}".format(
                        center_x=puddle.x + 0.5 * puddle.width,
                        center_y=puddle.y + 0.5 * puddle.height,
                        center_z=0.5 * puddle.depth,
                    ),
                    size="{half_width} {half_height} {half_depth}".format(
                        half_width=0.5 * puddle.width,
                        half_height=0.5 * puddle.height,
                        half_depth=0.5 * puddle.depth,
                    ),
                    conaffinity="1", # has contact with other objs
                    rgba="{r} {g} {b} {a}".format(
                        r=r, g=g, b=b, a=a
                    ),
                ))
            else:
                worldbody.append(ET('geom',
                    type="box",
                    pos="{center_x} {center_y} 0".format(
                        center_x=puddle.x + 0.5 * puddle.width,
                        center_y=puddle.y + 0.5 * puddle.height,
                    ),
                    size="{half_width} {half_height} 0.01".format(
                        half_width=0.5 * puddle.width,
                        half_height=0.5 * puddle.height,
                    ),
                    conaffinity="0", # no contact with other objs
                    rgba="{r} {g} {b} {a}".format(
                        r=r, g=g, b=b, a=a
                    ),
                ))
        if self.reward_type == "goal":
            worldbody.append(ET('geom',
                type="box",
                pos="{center_x} {center_y} 0".format(
                    center_x=self.goal[0],
                    center_y=self.goal[1],
                ),
                size="0.5 0.5 0.01",
                conaffinity="0", # no contact with other objs
                rgba="0 0 1 0.8"
            ))

        # output to file
        s = etree.tostring(root, pretty_print=True).decode()
        with open(self.file_path, "w") as f:
            f.write(s)
        # print(s)
        print("Generated the xml file to %s"%(self.file_path))


    def get_viewer(self, config=None):
        if self.viewer is None:
            self.viewer = MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer.cam.trackbodyid = 1
            self.viewer.cam.distance = 30
            self.viewer.cam.elevation = -70
        if config is not None:
            self.viewer.set_window_pose(config["xpos"], config["ypos"])
            self.viewer.set_window_size(config["width"], config["height"])
            self.viewer.set_window_title(config["title"])
        else:
            self.viewer.set_window_pose(1000,0)
            self.viewer.set_window_size(500, 500)
            self.viewer.set_window_title("ant puddle")
        return self.viewer

class AntPuddleGenerator(object):
    # env generating shortcuts
    def generate_u_shaped_maze(self, wall_offset, length, turn_length,
        obj=""):
        spacing = 2. + 2 * wall_offset
        puddles = [
            Puddle(x=-1, y=spacing/2, width=length-spacing+1, height=1,
                angle=0, cost=0, text="0",
                plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
            Puddle(x=-1, y=-1-spacing/2, width=length+1, height=1,
                angle=0, cost=0, text="1",
                plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
            Puddle(x=-1, y=turn_length-spacing/2-1, width=length-spacing+1, height=1,
                angle=0, cost=0, text="2",
                plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
            Puddle(x=-1, y=turn_length+spacing/2, width=length+1, height=1,
                angle=0, cost=0, text="3",
                plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),

            Puddle(x=length-spacing-1, y=spacing/2, width=1, height=turn_length-spacing,
                angle=0, cost=0, text="4",
                plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
            Puddle(x=length, y=-1-spacing/2, width=1, height=turn_length+spacing+2,
                angle=0, cost=0, text="5",
                plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
            Puddle(x=-2, y=-1-spacing/2, width=1, height=spacing+2,
                angle=0, cost=0, text="6",
                plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
            Puddle(x=-2, y=turn_length-spacing/2-1, width=1, height=spacing+2,
                angle=0, cost=0, text="7",
                plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
        ]
        goal = (0, turn_length)
        xmin = min([p.x for p in puddles])
        xmax = max([p.x + p.width for p in puddles])
        ymin = min([p.y for p in puddles])
        ymax = max([p.y + p.height for p in puddles])
        plot_offset = 0.5
        plot_settings = dict(
            xlim=(xmin - plot_offset, xmax + plot_offset),
            ylim=(ymin - plot_offset, ymax + plot_offset),
        )
        if obj == "":
            return puddles, goal, plot_settings
        else:
            return locals()[obj]


    def generate_two_choice_maze(self, wall_offset, length, obj="puddles"):
        spacing = 2. + 2 * wall_offset
        puddles = [
            Puddle(x=-spacing/2, y=-2-spacing, width=length+2*spacing, height=1,
                angle=0, cost=0, hard=True, text="0",
                plot_args=dict(color=(1., 0., 0., 1.0))),
            Puddle(x=-spacing/2, y=1+spacing, width=length+2*spacing, height=1,
                angle=0, cost=0, hard=True, text="1",
                plot_args=dict(color=(1., 0., 0., 1.0))),
            Puddle(x=-1-spacing/2, y=-2-spacing, width=1, height=4+2*spacing,
                angle=0, cost=0, hard=True, text="2",
                plot_args=dict(color=(1., 0., 0., 1.0))),
            Puddle(x=length+1.5*spacing, y=-2-spacing, width=1, height=4+2*spacing,
                angle=0, cost=0, hard=True, text="3",
                plot_args=dict(color=(1., 0., 0., 1.0))),
            Puddle(x=0.5+spacing/2, y=-0.5, width=length, height=1,
                angle=0, cost=0, hard=True, text="4",
                plot_args=dict(color=(1., 0., 0., 1.0))),
            Puddle(x=spacing/2+length, y=+0.5, width=0.5+spacing, height=1,
                angle=0, cost=0, hard=True, text="5",
                plot_args=dict(color=(1., 0., 0., 1.0))),
        ]
        goal = (0.5+spacing/2+length+spacing/2, 0)
        xmin = min([p.x for p in puddles])
        xmax = max([p.x + p.width for p in puddles])
        ymin = min([p.y for p in puddles])
        ymax = max([p.y + p.height for p in puddles])
        plot_offset = 0.5
        plot_settings = dict(
            xlim=(xmin - plot_offset, xmax + plot_offset),
            ylim=(ymin - plot_offset, ymax + plot_offset),
        )
        return locals()[obj]
