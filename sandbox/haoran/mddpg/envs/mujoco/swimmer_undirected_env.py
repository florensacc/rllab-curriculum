from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
from sandbox.haoran.mddpg.policies.mnn_policy import MNNPolicy
from sandbox.haoran.mddpg.policies.nn_policy import NNPolicy
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy
import os
import json
import numpy as np
import gc
import matplotlib.pyplot as plt


class SwimmerUndirectedEnv(MujocoEnv, Serializable):

    FILE = 'swimmer.xml'

    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            visitation_plot_config=None,
            prog_threshold=2.,
            motion_reward=True,
            visitation_reward=False,
            dist_reward=0,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.visitation_plot_config = visitation_plot_config
        self.prog_threshold = prog_threshold

        self.motion_reward = motion_reward
        self.visitation_reward = visitation_reward
        self.dist_reward = dist_reward

        self.visitation_bins = np.zeros((8, 12))
        self.visitation_reward_coeff = 1.
        self.visitation_leak_factor = 0.999

        super(SwimmerUndirectedEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())


    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            #self.get_body_com("torso").flat,
        ]).reshape(-1)

    def get_com(self):
        return self.get_body_com("torso").flat[:2]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        action_norm_sq = np.sum(np.square(action / scaling))
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * action_norm_sq
        if (np.abs(action / scaling) > 1).any():
            dist = max((np.abs(action / scaling) - 1).max(), 0)
            ctrl_cost += 1. * dist**2
            #ctrl_cost += 0.1


        done = False
        com = np.concatenate([self.get_body_com("torso").flat]).reshape(-1)

        motion_reward = (
            np.linalg.norm(self.get_body_comvel("torso"))
            if self.motion_reward else 0.0
        )

        visitation_reward = (
            self.get_visitation_reward(com) if self.visitation_reward else 0.0
        )
        distance_reward = self.dist_reward * (com[0] > self.prog_threshold)
        reward = motion_reward + visitation_reward - ctrl_cost \
            + distance_reward
            # send the com separately as env_info to avoid problems induced
            # by normalizing the observations

        return Step(next_obs, reward, done, com=com,
            distance_reward=distance_reward)

    def get_visitation_reward(self, com):
        th = np.arctan2(com[0], com[1]) + np.pi
        dist = np.linalg.norm(com)

        th_bin = int(th / np.pi * 4)
        if th_bin == 8:  # Hacky fix for the corner case.
            th_bin = 1
        dist_bin = int(dist / 1.0)

        self.visitation_bins[th_bin, dist_bin] += 1.
        self.visitation_bins *= self.visitation_leak_factor

        reward =  self.visitation_reward_coeff / (
            self.visitation_bins[th_bin, dist_bin] + 1.)

        return reward

    def log_stats(self, algo, epoch, paths):
        # forward distance
        progs = []
        for path in paths:
            coms = path["env_infos"]["com"]
            progs.append(coms[-1][0] - coms[0][0])
                # x-coord of com at the last time step minus the 1st step
        distance_rewards = np.concatenate([
            path["env_infos"]["distance_reward"]
            for path in paths
        ])
        n_directions = [
            np.max(progs) > self.prog_threshold,
            np.min(progs) < - self.prog_threshold,
        ].count(True)
        stats = {
            'env: ForwardProgressAverage': np.mean(progs),
            'env: ForwardProgressMax': np.max(progs),
            'env: ForwardProgressMin': np.min(progs),
            'env: ForwardProgressStd': np.std(progs),
            'env: n_directions': n_directions,
            'env: DistanceRewardAverage': np.mean(distance_rewards),
            'env: DistanceRewardMax': np.max(distance_rewards),
            'env: DistanceRewardMin': np.min(distance_rewards),
            'env: DistanceRewardStd': np.std(distance_rewards),
        }
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
        SwimmerUndirectedEnv.plot_paths(algo, paths, variant, img_file)

        return stats

    @staticmethod
    def plot_paths(paths, ax):
        for path in paths:
            pos = path['env_infos']['com']
            xx = pos[:, 0]
            yy = pos[:, 1]
            ax.plot(xx, yy, 'b')
        xlim = np.ceil(np.max(np.abs(xx)))
        ax.set_xlim((-xlim, xlim))
        ax.set_ylim((-2, 2))

    @staticmethod
    def eval_qf(sess, qf, o, lim):
        xx = np.arange(-lim, lim, 0.05)
        X, Y = np.meshgrid(xx, xx)
        all_actions = np.vstack([X.ravel(), Y.ravel()]).transpose()
        obs = np.array([o] * all_actions.shape[0])

        feed = {
            qf.observations_placeholder: obs,
            qf.actions_placeholder: all_actions
        }
        Q = sess.run(qf.output, feed).reshape(X.shape)
        return X, Y, Q

    @staticmethod
    def plot_paths(algo, paths, variant, img_file):
        # plot the test paths
        fig = plt.figure(figsize=(9, 14))
            # don't ask me how I figured out this ratio
        ax_paths = fig.add_subplot(211)
        ax_paths.grid(True)
        for path in paths:
            positions = path["env_infos"]["com"]
            xx = positions[:,0]
            yy = positions[:,1]
            ax_paths.plot(xx, yy, 'b')
        ax_paths.set_xlim((-4., 4.))
        ax_paths.set_ylim((-4., 4.))

        # plot the q-value at the initial state
        ax_qf = fig.add_subplot(212)
        ax_qf.grid(True)
        lim = 1.
        ax_qf.set_xlim((-lim, lim))
        ax_qf.set_ylim((-lim, lim))
        obs = paths[0]["observations"][0]
            # assume the initial state is fixed
            # assume the observations are not normalized
        X, Y, Q = SwimmerUndirectedEnv.eval_qf(algo.sess, algo.qf, obs, lim)
        if hasattr(algo, "alpha"):
            alpha = algo.alpha
        else:
            alpha = 1
        log_prob = (Q - np.max(Q.ravel())) / alpha
        ax_qf.clear()
        cs = ax_qf.contour(X, Y, log_prob, 20)
        ax_qf.clabel(cs, inline=1, fontsize=10, fmt='%.2f')

        # sample and plot actions
        if isinstance(algo.policy, StochasticNNPolicy):
            all_obs = np.array([obs] * algo.K)
            all_actions = algo.policy.get_actions(all_obs)[0]
        elif isinstance(algo.policy, MNNPolicy):
            all_actions, info = algo.policy.get_action(obs, k="all")
        elif isinstance(algo.policy, NNPolicy):
            all_actions, _ = algo.policy.get_action(obs)
        else:
            raise NotImplementedError

        x = all_actions[:, 0]
        y = all_actions[:, 1]
        ax_qf.plot(x, y, '*')

        # plot the action boundary, counterclockwise from the bottom left
        ax_qf.plot(
            [-1, 1, 1, -1, -1],
            [-1, -1, 1, 1, -1],
            'k-',
        )

        # title contains the variant parameters
        fig_title = variant["exp_name"] + "\n"
        for key in sorted(variant.keys()):
            fig_title += "%s: %s \n"%(key, variant[key])
        ax_paths.set_title(fig_title, multialignment="left")
        fig.tight_layout()

        plt.axis('equal')
        plt.draw()
        plt.pause(0.001)

        # save to file
        plt.savefig(img_file, dpi=100)
        plt.cla()
        plt.close('all')
        gc.collect()
