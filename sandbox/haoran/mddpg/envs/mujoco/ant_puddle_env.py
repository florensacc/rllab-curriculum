from sandbox.tuomas.mddpg.envs.mujoco.mujoco_env import MujocoEnv
from sandbox.haoran.myscripts.quaternion import Quaternion
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from rllab.core.serializable import Serializable
import numpy as np
import os
from matplotlib.patches import Rectangle

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

class Puddle(Serializable):
    """
    A puddle is a rectangular high-cost region
    """
    def __init__(self, x, y, width, height, angle, plot_args, cost):
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

    def plot(self, ax):
        ax.add_patch(Rectangle(
            xy=(self.x, self.y),
            width=self.width,
            height=self.height,
            angle=self.angle,
            **self.plot_args
        ))

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
            mujoco_env_args=dict()
        ):
        if direction is not None:
            assert np.isclose(np.linalg.norm(direction), 1.)
        self.puddles = puddles
        self.reward_type = reward_type
        self.direction = direction
        self.goal = goal
        self.flip_thr = flip_thr

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
        comvel = self.get_body_comvel("torso")
        if self.reward_type == "velocity":
            if self.direction is not None:
                motion_reward = comvel[0:2].dot(np.array(self.direction))
            else:
                motion_reward = np.linalg.norm(comvel[0:2])
        elif self.reward_type == "distance_from_origin":
            motion_reward = np.linalg.norm(self.get_body_com("torso")[:2])
        elif self.reward_type == "goal":
            pos = self.get_body_com("torso")[:2]
            motion_reward = np.linalg.norm(
                pos - np.array(self.goal)
            )
        else:
            raise NotImplementedError

        state = self._state

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5

        action_violations = np.maximum(np.maximum(lb - action, action - ub), 0)
        action_violation_cost = np.sum((action_violations / scaling)**2)

        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
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

    @overrides
    def plot_paths(self, paths, ax):
        ax.grid(True)
        self.plot_puddles(ax)
        if self.reward_type == "goal":
            self.plot_goal(ax)
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


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 20
