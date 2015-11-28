import tensorfuse as theano
import tensorfuse.tensor as TT
import numpy as np
from rllab.mdp.base import SymbolicMDP
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, merge_dict
from rllab.misc.viewer2d import Viewer2D, Colors


# code adapted from John's control repo
class CartpoleMDP(SymbolicMDP, Serializable):
    def __init__(self, horizon=100):
        self.max_pole_angle = .2
        self.max_cart_pos = 2.4
        self.max_cart_speed = 4.
        self.max_pole_speed = 4.
        self.max_force = 10.
        self.dt = .05
        self._viewer = None
        super(CartpoleMDP, self).__init__(horizon)
        Serializable.__init__(self, horizon)

    def reset_sym(self):
        bounds = np.array([
            self.max_cart_pos,
            self.max_cart_speed,
            self.max_pole_angle,
            self.max_pole_speed
        ])
        low, high = -0.05*bounds, 0.05*bounds
        state = theano.random.uniform(size=(4,), low=low, high=high, ndim=1)
        obs = state
        return state, obs

    @property
    @overrides
    def state_bounds(self):
        bounds = np.array([
            self.max_cart_pos,
            self.max_cart_speed,
            self.max_pole_angle,
            self.max_pole_speed
        ])
        return -bounds, bounds

    @property
    @overrides
    def action_bounds(self):
        bounds = np.array([1.])#self.max_force]) 
        return -bounds, bounds

    @property
    @overrides
    def observation_shape(self):
        return (4,)

    @property
    @overrides
    def state_shape(self):
        return (4,)

    @property
    @overrides
    def observation_dtype(self):
        return 'float32'

    @property
    @overrides
    def action_dim(self):
        return 1

    @property
    @overrides
    def action_dtype(self):
        return 'float32'

    @overrides
    def reward_sym(self, state, action):
        u, th = extract(
            self._decode(state, action),
            "u", "th"
        )
        ucost = 1e-5*(u**2).sum(axis=u.ndim-1)
        xcost = 1-TT.cos(th)
        done = self.done_sym(state)
        notdone = 1.-TT.cast(done, 'float32')
        reward = notdone*10 - notdone*xcost - notdone*ucost
        return reward

    def cost_sym(self, state, action):
        u, th = extract(
            self._decode(state, action),
            "u", "th"
        )
        ucost = 1e-5*(u**2).sum(axis=u.ndim-1)
        xcost = 1-TT.cos(th)
        return xcost + ucost

    def final_cost_sym(self, state):
        return TT.constant(0)

    @overrides
    def observation_sym(self, state):
        return state

    def _decode(self, state, action=None):
        x = state
        z = TT.take(x,0,axis=x.ndim-1)
        zdot = TT.take(x,1,axis=x.ndim-1)    
        th = TT.take(x,2,axis=x.ndim-1)
        thdot = TT.take(x,3,axis=x.ndim-1)
        ret = dict(z=z, zdot=zdot, th=th, thdot=thdot)
        if action:
            u = TT.clip(action, -1., 1.) * self.max_force
            u0 = TT.take(u,0,axis=u.ndim-1)
            return merge_dict(ret, dict(u=u, u0=u0))
        else:
            return ret

    @overrides
    def forward_sym(self, state, action):
        z, zdot, th, thdot, u0 = extract(
            self._decode(state, action),
            "z", "zdot", "th", "thdot", "u0"
        )

        dt = self.dt

        th1 = np.pi - th

        g = 10.
        mc = 1. # mass of cart
        mp = .1 # mass of pole
        muc = .0005 # coeff friction of cart
        mup = .000002 # coeff friction of pole
        l = 1. # length of pole

        def sign(x):
            return TT.switch(x>0, 1., -1.)

        thddot = -(-g*TT.sin(th1)
         + TT.cos(th1) * (-u0 - mp * l *thdot**2 * TT.sin(th1) + muc*sign(zdot))/(mc+mp)
          - mup*thdot / (mp*l))  \
        / (l*(4/3. - mp*TT.cos(th1)**2 / (mc + mp)))
        zddot = (u0 + mp*l*(thdot**2 * TT.sin(th1) - thddot * TT.cos(th1)) - muc*sign(zdot))  \
            / (mc+mp)
        newzdot = zdot + dt*zddot
        newz = z + dt*newzdot
        newthdot = thdot + dt*thddot
        newth = th + dt*newthdot

        return TT.stack([newz, newzdot, newth, newthdot]).T

    @overrides
    def done_sym(self, x):
        z, th = extract(
            self._decode(x),
            "z", "th"
        )
        z = TT.take(x, 0, axis=x.ndim-1)
        th = TT.take(x, 2, axis=x.ndim-1)
        return (z > self.max_cart_pos) | (z < -self.max_cart_pos) | (th > self.max_pole_angle) | (th < -self.max_pole_angle) 

    def plot(self, states=None, actions=None, pause=False):

        if states is None:
            states = [self.state]
        if actions is None:
            actions = [self.action]

        if self._viewer is None:
            self._viewer = Viewer2D(size=(640, 480), xlim=[-5,5], ylim=[-5.0/640*480,5.0/640*480])

        viewer = self._viewer
        viewer.fill(Colors.white)

        def draw_cart(x, alpha=255):
            pole_width = 10 * 10.0 / 640
            pole_height = 7.5 * 100.0 / 480
            cart_width = 10 * 50.0 / 640
            cart_height = 7.5 * 30.0 / 480

            cart_size = np.array([cart_width, cart_height])
            cart_pos = np.array([x[0], 0])

            poleang = x[2]
            #if u:
            #    force = u[0]
            
            viewer.rect((0, 0, 0, alpha), cart_pos, cart_size)
            pole_origin = np.array([x[0], cart_height / 2])
            pole_local_points = np.array([[-pole_width/2, 0], [pole_width/2, 0], [pole_width/2, pole_height], [-pole_width/2, pole_height]])
            pole_rotmat = np.array([[np.cos(poleang),-np.sin(poleang)],[np.sin(poleang),np.cos(poleang)]])
            viewer.polygon((0, 255, 0, alpha), pole_origin[None, :] + pole_local_points.dot(pole_rotmat))
            
            #if u:
            #    viewer.line(Colors.red, cart_pos, (x[0] + force, 0), width=0.1)
        if len(states) <= 2:
            map(draw_cart, states)
        else:
            draw_cart(states[0])
            for i in range(1, len(states) - 1):
                draw_cart(states[i], 30)
            draw_cart(states[-1])
        viewer.loop_once()
        if pause:
            viewer.pause()
