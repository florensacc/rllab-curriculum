# Taken from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
import theano.tensor as TT
import numpy as np

from rllab.envs.base import Env, Step
from rllab.misc import ext
from rllab import spaces
from rllab.misc.overrides import overrides
from rllab.misc import logger

class AnalyticCartpoleEnv(Env):
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        # Postition limit of the cart
        self.x_threshold = 2.4
        self.force_threshold = 10.0
        # Reward received for staying alive
        self.alive_coeff = 1.0

        self.init_sym()
        self.reset()

    @property
    @overrides
    def observation_space(self):
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        ub = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
            )
        lb = -ub
        return spaces.Box(lb,ub)

    @property
    @overrides
    def action_space(self):
        ub = np.asarray([self.force_threshold])
        lb = -ub
        return spaces.Box(lb,ub)

    def init_sym(self):
        logger.log("Compiling analytic derivatives for the environment.")
        # dynamics
        s = TT.fvector('s')
        a = TT.fvector('a')
        x = s[0]
        x_dot = s[1]
        theta = s[2]
        theta_dot = s[3]

        costheta = TT.cos(theta)
        sintheta = TT.sin(theta)
        force = a[0]
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x2  = x + self.tau * x_dot
        x2_dot = x_dot + self.tau * xacc
        theta2 = theta + self.tau * theta_dot
        theta2_dot = theta_dot + self.tau * thetaacc

        s2_list = [x2,x2_dot,theta2,theta2_dot]
        s2 = TT.stack(s2_list)
        ds2_ds = TT.stack(
            [ TT.grad(s2_i,s) for s2_i in s2_list ],
            axis=0
        )
        ds2_da = TT.stack(
            [
                TT.grad(
                    s2_i,
                    wrt=a,
                    disconnected_inputs='ignore',
                    return_disconnected='zero'
                )
            for s2_i in s2_list],
            # has to ignore disconnected inputs because actions do not contribute to x,theta update directly
            axis=0
        )

        self.f = ext.compile_function(
            inputs = [s,a],
            outputs = s2,
        )
        self.f_s = ext.compile_function(
            inputs = [s,a],
            outputs = ds2_ds,
        )
        self.f_a = ext.compile_function(
            inputs = [s,a],
            outputs = ds2_da,
        )

        # termination (make it float instead of bool)
        notdone =  (x >= -self.x_threshold) * \
                   (x <= self.x_threshold) * \
                   (theta >= -self.theta_threshold_radians) * \
                   (theta <= self.theta_threshold_radians)
        done = 1 - notdone
        self.done = ext.compile_function(
            inputs = [s,a],
            outputs = done,
        )

        # rewards
        a_cost = 1e-5 * (force**2)
        s2_cost = 1 - TT.cos(theta2)
        r = notdone * (self.alive_coeff - a_cost - s2_cost)

        self.r = ext.compile_function(
            inputs = [s,a],
            outputs = r
        )

        dr_ds = TT.grad(r,s,disconnected_inputs='ignore')
        dr_da = TT.grad(r,a,disconnected_inputs='ignore')
        self.r_s = ext.compile_function(
            inputs = [s,a],
            outputs = dr_ds,
        )
        self.r_a = ext.compile_function(
            inputs = [s,a],
            outputs = dr_da,
        )

    def step(self,action):
        state = self.state
        next_state = self.f(state,action)
        reward = self.r(state,action)
        done = self.done(state,action)
        self.state = next_state
        return Step(next_state,reward,done)

    def get_current_obs(self):
        return self.state

    def log_diagnostics(self,paths):
        path_lens = [
            len(path["rewards"])
            for path in paths
        ]
        logger.record_tabular_misc_stat("PathLength",path_lens)

    def reset(self,init_state=None):
        if init_state is not None:
            self.state = np.asarray(init_state)
        else:
            self.state = np.random.uniform(low=-0.05,high=0.05,size=(4,))
            # test whether termination is implemented correctly
            self.state = np.asarray([0,0,0,0])
        return self.state
