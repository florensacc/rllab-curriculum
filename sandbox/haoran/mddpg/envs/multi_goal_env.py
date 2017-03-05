from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.envs.base import Env
from rllab.spaces.box import Box
from sandbox.haoran.mddpg.policies.mnn_policy import MNNPolicy
from sandbox.haoran.mddpg.policies.nn_policy import NNPolicy
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy

import os
import gc
import json
import numpy as np
import matplotlib.pyplot as plt

class MultiGoalEnv(Env, Serializable):
    """
    Moving a 2D point mass to one of the goal positions.
    Cost is the distance to the shortest goal.

    state: position
    action: velocity
    """
    def __init__(self, goal_reward=0):
        Serializable.quick_init(self, locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.array((0,0), dtype=np.float32)
        self.init_sigma = 0
        self.goal_positions = np.array(
            [
                [5, 0],
                [-5, 0],
                [0, 5],
                [0, -5]
            ],
            dtype=np.float32
        )
        self.goal_threshold = 1.
        self.goal_reward = goal_reward
        self.action_cost_coeff = 0
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1
        self.reset()
        self.dynamic_plots = []

    def reset(self):
        unclipped_observation = self.init_mu + self.init_sigma * \
            np.random.normal(size=(self.dynamics.s_dim))
        o_lb, o_ub = self.observation_space.bounds
        self.observation = np.clip(unclipped_observation, o_lb, o_ub)
        return self.observation

    @property
    def observation_space(self):
        return Box(
            low=np.array((self.xlim[0],self.ylim[0])),
            high=np.array((self.xlim[1],self.ylim[1])),
            shape=None
        )

    @property
    def action_space(self):
        return Box(
            low= -self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,)
        )

    def get_current_obs(self):
        return np.copy(self.observation)

    def plot_paths(self, paths, ax):
        self.plot_position_cost(ax)

        for path in paths:
            xx = path["observations"][:,0]
            yy = path["observations"][:,1]
            ax.plot(xx, yy, 'b-')

    def compute_log_barrier_cost(self, positions, barrier_position,
        radius, cutoff_dist=0.05, threshold=1e-5):
        """
        A bit different from Thomas' implementation. There is no
            cone-like cost inside the barrier to give gradient for
            traj-opt.
        :param positions: an N x 2 array
        """
        barrier = np.expand_dims(barrier_position,axis=0)
        dists = np.sqrt(np.sum(
            (positions - barrier)**2,
            axis=1
        ))
        dists_to_barrier = dists-radius
        induce_cost = (dists_to_barrier < cutoff_dist).astype(int)
        costs = - induce_cost * self.lam_barrier * \
            np.log(np.maximum(dists_to_barrier / cutoff_dist, threshold))
        return costs

    def step(self, action):
        a_lb, a_ub = self.action_space.bounds
        action = np.clip(action, a_lb, a_ub).ravel()
            # need to ravel because sometimes the input action has
            # shape (1,2) with tensorflow

        next_obs = self.dynamics.forward(self.observation, action)
        o_lb, o_ub = self.observation_space.bounds
        next_obs = np.clip(next_obs, o_lb, o_ub)

        reward = self.compute_reward(self.observation, action)
        cur_position = self.observation
        dist_to_goal = np.amin([
            np.linalg.norm(cur_position - goal_position)
            for goal_position in self.goal_positions
        ])
        done =  dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        self.observation = np.copy(next_obs)
        return next_obs, reward, done, {}

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation
        goal_cost = np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward

    def render(self,close=False):
        if not hasattr(self,'fig') or self.fig is None:
            self.fig = plt.figure()
            plt.axis('equal')
            self.ax = self.fig.add_subplot(111)
        if not hasattr(self,'fixed_plots') or self.fixed_plots is None:
            self.fixed_plots = self.plot_position_cost(self.ax)
        for obj in self.dynamic_plots:
            obj.remove()
        x,y = self.observation
        point = self.ax.plot(x,y,'b*')
        self.dynamic_plots = point

        plt.draw()
        plt.pause(0.001)
        if close:
            self.fixed_plots = None
            self.fig = None


    def plot_position_cost(self,ax):
        delta = 0.01
        xmin, xmax = tuple(1.1 * np.array(self.xlim))
        ymin, ymax = tuple(1.1 * np.array(self.ylim))
        X,Y = np.meshgrid(
            np.arange(xmin,xmax,delta),
            np.arange(ymin,ymax,delta)
        )
        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        positions = np.vstack([X.ravel(), Y.ravel()]).transpose()
        costs = goal_costs

        contours = ax.contour(X,Y,costs,20)
        ax.clabel(contours,inline=1,fontsize=10,fmt='%.0f')
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        goal = ax.plot(self.goal_positions[:,0],self.goal_positions[:,1],'ro')
        return [contours, goal]

    @overrides
    def log_diagnostics(self, paths):
        pass

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def log_stats(self, algo, epoch, paths):
        # compute number of goals reached
        n_goal = len(self.goal_positions)
        goal_reached = [False] * n_goal

        for path in paths:
            last_obs = path["observations"][-1]
            for i, goal in enumerate(self.goal_positions):
                if np.linalg.norm(last_obs - goal) < self.goal_threshold:
                    goal_reached[i] = True

        stats = {
            "env:goal_reached": goal_reached.count(True)
        }

        # plot q-values at selected states
        # snapshot_gap = logger.get_snapshot_gap()
        # if snapshot_gap <= 0 or \
        #     np.mod(epoch + 1, snapshot_gap) == 0 or \
        #     epoch == 0:

        snapshot_dir = logger.get_snapshot_dir()
        variant_file = os.path.join(
            snapshot_dir,
            "variant.json",
        )
        with open(variant_file) as vf:
            variant = json.load(vf)
        img_file = os.path.join(
            snapshot_dir,
            "itr_%d_qf.png"%(epoch),
        )
        self.plot_qf(algo, variant, img_file)
        return stats

    def eval_qf(self, sess, qf, o, lim):
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

    def plot_qf(self, algo, variant, img_file):
        lim = 2.

        # Set up all critic plots.
        critic_fig = plt.figure(figsize=(20, 7))
        ax_critics = []
        for i in range(3):
            ax = critic_fig.add_subplot(130 + i + 1)
            ax_critics.append(ax)
            plt.axis('equal')
            ax.set_xlim((-lim, lim))
            ax.set_ylim((-lim, lim))

        obss = np.array([[-2.5, 0.0],
                         [0.0, 0.0],
                         [2.5, 2.5]])

        for ax_critic, obs in zip(ax_critics, obss):

            X, Y, Q = self.eval_qf(algo.sess, algo.qf, obs, lim)

            ax_critic.clear()
            cs = ax_critic.contour(X, Y, Q, 20)
            ax_critic.clabel(cs, inline=1, fontsize=10, fmt='%.0f')

            # sample and plot actions
            if isinstance(algo.policy, StochasticNNPolicy):
                all_obs = np.array([obs] * algo.K)
                all_actions = algo.policy.get_actions(all_obs)[0]
            elif isinstance(algo.policy, MNNPolicy):
                all_actions, info = algo.policy.get_action(obs, k="all")
            elif isinstance(algo.policy, NNPolicy):
                all_actions, info = algo.policy.get_action(obs)
            else:
                raise NotImplementedError

            x = all_actions[:, 0]
            y = all_actions[:, 1]
            ax_critic.plot(x, y, '*')

            # plot the boundary, counterclockwise from the bottom left
            ax_critic.plot(
                [-1, 1, 1, -1, -1],
                [-1, -1, 1, 1, -1],
                'k-',
            )

        # write down the hyperparams in the title of the first axis
        fig_title = variant["exp_name"] + "\n"
        for key in sorted(variant.keys()):
            fig_title += "%s: %s \n"%(key, variant[key])

        # other axes will be the observation point
        for i in range(len(ax_critics)):
            ax_title = "state: (%.2f, %.2f)"%(obss[i][0], obss[i][1])
            if i == 0:
                ax_title = fig_title + ax_title
            ax_critics[i].set_title(ax_title, multialignment="left")
        critic_fig.tight_layout()

        # save to file
        plt.savefig(img_file, dpi=100)
        plt.cla()
        plt.close('all')
        gc.collect()


class PointDynamics(object):
    """
    state: (position, velocity)
    action: acceleration
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            np.random.normal(size=self.s_dim)
        return state_next
