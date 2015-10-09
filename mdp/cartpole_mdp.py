import theano
import theano.tensor as T
import numpy as np
from .base import MDP

class CartpoleMDP(MDP):
    def __init__(self):
        self.max_pole_angle = .2

        self.max_cart_pos = 2.4
        self.max_cart_speed = 4.
        self.max_pole_speed = 4.
        self.max_force = 10.
        self.dt = .05

    def sample_initial_states(self, n):
        bounds = np.tile(
            np.array([
                self.max_cart_speed,
                self.max_cart_speed,
                self.max_pole_speed,
                self.max_pole_speed
            ]).reshape(1, -1), (n, 1)
        )

        low, high = -0.05*bounds, 0.05*bounds

        states = np.random.uniform(low, high)
        obs = states
        return states, obs

    @property
    def observation_shape(self):
        return (4,)

    @property
    def n_actions(self):
        return 1

    def step(self, states, actions):

        states = np.array(states)
        actions = np.clip(actions, -self.max_force, self.max_force) #pylint: disable=E1111
        
        dt = self.dt

        z, zdot, th, thdot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        u0 = actions[:, 0]

        th1 = np.pi - th

        g = 10.
        mc = 1. # mass of cart
        mp = .1 # mass of pole
        muc = .0005 # coeff friction of cart
        mup = .000002 # coeff friction of pole
        l = 1. # length of pole

        thddot = -(-g*np.sin(th1)
         + np.cos(th1) * (-u0 - mp * l *thdot**2 * np.sin(th1) + muc*np.sign(zdot))/(mc+mp)
          - mup*thdot / (mp*l))  \
        / (l*(4/3. - mp*np.cos(th1)**2 / (mc + mp)))
        zddot = (u0 + mp*l*(thdot**2 * np.sin(th1) - thddot * np.cos(th1)) - muc*np.sign(zdot))  \
            / (mc+mp)

        newzdot = zdot + dt*zddot
        newz = z + dt*newzdot
        newthdot = thdot + dt*thddot
        newth = th + dt*newthdot

        done = (z > self.max_cart_pos) | (z < -self.max_cart_pos) | (th > self.max_pole_angle) | (th < -self.max_pole_angle) 

        ucost = 1e-5 * (u0**2)
        xcost = 1 - np.cos(th)
        notdone = 1 - done

        next_states = np.array([newz, newzdot, newth, newthdot]).T
        next_obs = next_states
        rewards = notdone*10 - notdone*xcost - notdone*ucost

        return next_states, next_obs, rewards, done, np.ones_like(done)
