from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import os
import numpy as np
import json
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import gc

class MultiGoalReacherEnv(MujocoEnv, Serializable):
    """
    Same as Reacher, but with the cost defined as the distance to the closest
        goal.
    Mujoco only renders one goal, since modifying the xml is annoying.
    Observation modified:
        1. only observe angles and linear velocities of the two joints
        2. no longer observing the goal position
        3. no longer knowing the relative position of the goal to fingertip
    """
    FILE = "reacher.xml"

    def __init__(
            self,
            deterministic=False,
            goal_threshold=0.03,
            goals=[
                [0., 0.19],
                [0., -0.19],
                [0.19, 0.],
                [-0.19, 0.],
                [0.134, 0.134],
                [-0.134, 0.134],
                [0.134, -0.134],
                [-0.134, -0.134],
            ],
            use_sincos=True,
            goal_reward=0.,
            *args,
            **kwargs
        ):
        """
        :param goal_threshold: if the dist to goal is under the threshold,
            terminate
        :param deterministic: fix the initial reacher position and velocity and
            the goal position
        """
        self.frame_skip = 2 # default in OpenAI
        self.deterministic = deterministic
        self.goal_threshold = goal_threshold
        self.goal_positions = np.array(goals)
        self.use_sincos = use_sincos
        self.goal_reward = goal_reward
        super(MultiGoalReacherEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def plot_env(self, ax):
        for pos in self.goal_positions:
            ax.plot(pos[0], pos[1], 'xk', mew=8, ms=16)

    @overrides
    def render(self,close=False):
        viewer = self.get_viewer()
        viewer.cam.trackbodyid = 0
        viewer.loop_once()
        if close:
            self.stop_viewer()

    def step(self, action):
        fingertip_pos_2d = self.get_body_com("fingertip")[:2]
        dists = [np.linalg.norm(fingertip_pos_2d - goal) for goal in self.goal_positions]
        min_dist = np.amin(dists)
        reward_dist = - min_dist
        reward_ctrl = - np.square(action).sum()
        self.forward_dynamics(action)
        # note: mujoco_env automatically clips the actions
        ob = self.get_current_obs()
        done = min_dist < self.goal_threshold
        reward_goal = done * self.goal_reward
        reward = reward_dist + reward_ctrl + reward_goal
        env_info = dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            fingertip_pos_2d=fingertip_pos_2d,
        )
        # FOR TESTING:
        # if done:
        #     i = np.argmin(dists)
        #     print("Reached goal:", self.goal_positions[i])
        return ob, reward, done, env_info

    @overrides
    def reset_mujoco(self, init_state=None, qpos_init=None):
        # For now, just fix the initial state.
        qpos = np.array([0, 3.0, .1, -.1])
        if qpos_init is not None:
            qpos[:2] = qpos_init

        qvel = np.array([0, 0, 0, 0])
        qacc = np.array([0, 0, 0, 0])
        ctrl = np.array([0, 0])

        self.model.data.qpos = qpos[:, None]
        self.model.data.qvel = qvel[:, None]
        self.model.data.qacc = qacc[:, None]
        self.model.data.ctrl = ctrl[:, None]

        return self.get_current_obs()
        # if self.deterministic:
        #     qpos = np.copy(self.init_qpos)
        #     qpos[1,0] = 3. # the initial fingertip is near origin
        # else:
        #     qpos = self.init_qpos + np.random.uniform(
        #         low=-0.1, high=0.1, size=(self.model.nq, 1))
        #
        # # render the first goal
        # qpos[-2:] = np.array([[self.goal_positions[0,0]], [self.goal_positions[0,1]]])
        # if self.deterministic:
        #     qvel = np.copy(self.init_qvel)
        # else:
        #     qvel = self.init_qvel + \
        #         np.random.uniform(low=-.005, high=.005, size=(self.model.nv, 1))
        # # the goal should not move no matter what
        # qvel[-2:] = 0
        # self.model.data.qpos = qpos
        # self.model.data.qvel = qvel
        # self.model.data.qacc = self.init_qacc
        # self.model.data.ctrl = self.init_ctrl
        # return self.get_current_obs()

    def get_current_obs(self):
        if self.use_sincos:
            theta = self.model.data.qpos.flat[:2] # joint angles
            return np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.model.data.qvel.flat[:2],
            ])
        else:
            return np.concatenate([
                self.model.data.qpos.flat[:2],
                self.model.data.qvel.flat[:2],
            ])

    def log_stats(self, algo, epoch, paths):
        # log the number of goals reached
        n_goal = len(self.goal_positions)
        goal_reached = [False] * n_goal
        for path in paths:
            fingertip_pos_2d = path["env_infos"]["fingertip_pos_2d"][-1]
            for i, goal in enumerate(self.goal_positions):
                if np.linalg.norm(fingertip_pos_2d - goal) < self.goal_threshold:
                    goal_reached[i] = True
                if np.all(goal_reached):
                    break
        stats = {
            "env:goal_reached": goal_reached.count(True)
        }

        # if it is time to take snapshots, plot the test paths
        snapshot_gap = logger.get_snapshot_gap()
        if snapshot_gap <= 0 or \
            np.mod(epoch + 1, snapshot_gap) == 0 or \
            epoch == 0:
            snapshot_dir = logger.get_snapshot_dir()
            variant_file = os.path.join(
                snapshot_dir,
                "variant.json",
            )
            with open(variant_file) as vf:
                variant = json.load(vf)
            img_file = os.path.join(
                snapshot_dir,
                "itr_%d_test_paths.png"%(epoch),
            )
            self.plot_paths(paths, variant, img_file, epoch)
        return stats

    def plot_paths(self, paths, variant, img_file, epoch):
        # plot the test paths
        fig = plt.figure(figsize=(7,7))
        ax_paths = fig.add_subplot(111)
        self.plot_env(ax_paths)
        plt.axis('equal')
        ax_paths.grid(True)
        ax_paths.set_xlim((-0.3, 0.3))
        ax_paths.set_ylim((-0.3, 0.3))
        for path in paths:
            positions = path["env_infos"]["fingertip_pos_2d"] # T x 2
            xx = positions[:,0]
            yy = positions[:,1]
            ax_paths.plot(xx, yy, 'b')
        plt.draw()
        plt.pause(0.001)

        # title contains the variant parameters
        fig_title = variant["exp_name"] + "\n" + "epoch: %d"%(epoch) + "\n"
        for key in sorted(variant.keys()):
            fig_title += "%s: %s \n"%(key, variant[key])
        ax_paths.set_title(fig_title, multialignment="left")
        fig.tight_layout()

        # save to file
        plt.savefig(img_file, dpi=100)
        plt.cla()
        plt.close('all')
        gc.collect()
