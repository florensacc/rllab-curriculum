import cgtcompat as theano
import cgtcompat.tensor as T
import numpy as np
from .base import SymbolicMDP
from misc.overrides import overrides

# code adapted from John's control repo
class CartpoleMDP(SymbolicMDP):
    def __init__(self):
        self.max_pole_angle = .2
        self.max_cart_pos = 2.4
        self.max_cart_speed = 4.
        self.max_pole_speed = 4.
        self.max_force = 10.
        self.dt = .05

    def reset(self):
        bounds = np.array([
            self.max_cart_speed,
            self.max_cart_speed,
            self.max_pole_speed,
            self.max_pole_speed
        ])
        low, high = -0.05*bounds, 0.05*bounds
        state = np.random.uniform(low, high)
        obs = state
        return state, obs

    @property
    def observation_shape(self):
        return (4,)

    @property
    def n_actions(self):
        return 1

    @overrides
    def step_symbolic(self, state, action):

        u = T.clip(action, -self.max_force, self.max_force)
        x = state

        dt = self.dt

        z = T.take(x,0,axis=x.ndim-1)
        zdot = T.take(x,1,axis=x.ndim-1)    
        th = T.take(x,2,axis=x.ndim-1)
        thdot = T.take(x,3,axis=x.ndim-1)
        u0 = T.take(u,0,axis=u.ndim-1)

        th1 = np.pi - th

        g = 10.
        mc = 1. # mass of cart
        mp = .1 # mass of pole
        muc = .0005 # coeff friction of cart
        mup = .000002 # coeff friction of pole
        l = 1. # length of pole

        def sign(x):
            return T.switch(x>0, 1, -1)

        thddot = -(-g*T.sin(th1)
         + T.cos(th1) * (-u0 - mp * l *thdot**2 * T.sin(th1) + muc*sign(zdot))/(mc+mp)
          - mup*thdot / (mp*l))  \
        / (l*(4/3. - mp*T.cos(th1)**2 / (mc + mp)))
        zddot = (u0 + mp*l*(thdot**2 * T.sin(th1) - thddot * T.cos(th1)) - muc*sign(zdot))  \
            / (mc+mp)

        newzdot = zdot + dt*zddot
        newz = z + dt*newzdot
        newthdot = thdot + dt*thddot
        newth = th + dt*newthdot

        done = (z > self.max_cart_pos) | (z < -self.max_cart_pos) | (th > self.max_pole_angle) | (th < -self.max_pole_angle) 

        ucost = 1e-5*(u**2).sum(axis=u.ndim-1)
        xcost = 1-T.cos(th)
        notdone = 1-done

        reward = notdone*10 - notdone*xcost - notdone*ucost

        newx = T.stack([newz, newzdot, newth, newthdot]).T

        return newx, newx, reward, done
