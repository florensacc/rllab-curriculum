from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.envs.base import Env
from rllab.spaces.box import Box

import numpy as np
import matplotlib.pyplot as plt

class DoubleSlitEnvV2(Env, Serializable):
    """
    Moving a 2D point mass to a target location.
    Three circular log barriers form two slits for the mass to go through.
    A reward is given at each time step to encourage going to the goal and
        avoiding the barriers.
    We could modify the dynamics to make the barriers solid, but that makes
        it a bit hard to compare to traj-opt results.

    state: position
    action: velocity
    """
    def __init__(self):
        Serializable.quick_init(self, locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.array((0,0), dtype=np.float32)
        self.init_sigma = 0
        self.goal_position = np.array((5, 0), dtype=np.float32)
        self.goal_threshold = 1.
        self.goal_reward = 100
        self.barrier_info = ((np.array((3,  4)), 0.5),
                         (np.array((3,  0)), 0.5),
                         (np.array((3, -4)), 0.5))
        self.lam_barrier=30
        self.action_cost_coeff = 0
        self.xlim = (-2, 6)
        self.ylim = (-4, 4)
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
        dist_to_goal = np.linalg.norm(cur_position - self.goal_position)
        done =  dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        self.observation = np.copy(next_obs)
        return next_obs, reward, done, {}

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_pos = observation
        goal_cost = np.sum((cur_pos - self.goal_position)**2)

        # penalize staying with the log barriers
        barrier_cost = np.sum([
            self.compute_log_barrier_cost(np.array([cur_pos]),
                barrier_position, radius)[0]
            for barrier_position, radius in self.barrier_info
        ])
        costs = [action_cost, goal_cost, barrier_cost]
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

        if close:
            self.fixed_plots = None

    def plot_position_cost(self,ax):
        delta = 0.01
        xmin, xmax = tuple(1.1 * np.array(self.xlim))
        ymin, ymax = tuple(1.1 * np.array(self.ylim))
        X,Y = np.meshgrid(
            np.arange(xmin,xmax,delta),
            np.arange(ymin,ymax,delta)
        )
        goal_x, goal_y = self.goal_position
        goal_costs = (X - goal_x) ** 2 + (Y - goal_y) ** 2
        positions = np.vstack([X.ravel(), Y.ravel()]).transpose()
        barrier_costs = np.sum([
            self.compute_log_barrier_cost(
                positions, barrier_position, radius
            )
            for barrier_position, radius in self.barrier_info
        ],axis=0)
        barrier_costs = barrier_costs.reshape(X.shape)
        costs = goal_costs + barrier_costs

        contours = ax.contour(X,Y,costs,20)
        ax.clabel(contours,inline=1,fontsize=10,fmt='%.0f')
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        goal = ax.plot(goal_x,goal_y,'ro')
        return [contours, goal]

    @overrides
    def log_diagnostics(self, paths):
        pass

    def get_param_values(self):
        return None


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
