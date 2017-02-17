from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import os
import numpy as np

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
        reward = reward_dist + reward_ctrl
        self.forward_dynamics(action)
        ob = self.get_current_obs()
        done = min_dist < self.goal_threshold
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


        # Haoran's version below.
        if self.deterministic:
            qpos = np.copy(self.init_qpos)
            qpos[1,0] = 3. # the initial fingertip is near origin
        else:
            qpos = self.init_qpos + np.random.uniform(
                low=-0.1, high=0.1, size=(self.model.nq, 1))

        # render the first goal
        qpos[-2:] = np.array([[self.goal_positions[0,0]], [self.goal_positions[0,1]]])
        if self.deterministic:
            qvel = np.copy(self.init_qvel)
        else:
            qvel = self.init_qvel + \
                np.random.uniform(low=-.005, high=.005, size=(self.model.nv, 1))
        # the goal should not move no matter what
        qvel[-2:] = 0
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        import pdb; pdb.set_trace()
        return self.get_current_obs()

    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2] # joint angles
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
        ])

    def log_stats(self, epoch, paths):
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
        return stats
