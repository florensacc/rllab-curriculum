import tensorfuse as theano
import tensorfuse.tensor as TT
import numpy as np
from .base import SymbolicMDP
from misc.overrides import overrides
from misc.ext import extract, merge_dict

# code adapted from John's control repo
class CartpoleMDP(SymbolicMDP):
    def __init__(self, horizon=100):
        self.max_pole_angle = .2
        self.max_cart_pos = 2.4
        self.max_cart_speed = 4.
        self.max_pole_speed = 4.
        self.max_force = 10.
        self.dt = .05
        self.viewer = None
        super(CartpoleMDP, self).__init__(horizon)

    def reset_sym(self):
        bounds = np.array([
            self.max_cart_speed,
            self.max_cart_speed,
            self.max_pole_speed,
            self.max_pole_speed
        ])
        low, high = -0.05*bounds, 0.05*bounds
        state = theano.random.uniform(size=(4,), low=low, high=high, ndim=1)
        obs = state
        return state, obs

    @property
    def observation_shape(self):
        return (4,)

    @property
    def state_shape(self):
        return (4,)

    @property
    def n_actions(self):
        return 1

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
            u = TT.clip(action, -self.max_force, self.max_force)
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

    def plot(self):
        x = self.state
        u = self.action

        from misc.pygameviewer import PygameViewer, pygame
        if self.viewer is None:
            self.viewer = PygameViewer()
        screen = self.viewer.screen
        screen.fill((255,255,255))
        
        world_width = 10
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        scale = screen_width/world_width
        
        cartpos = x[0]*scale+screen_width/2
        

        poleang = x[2]
        if u:
            force = u[0]
        
        poleheight = 100
        polewidth=10
                
        cartwidth=50
        cartheight=30

        cartleftx = cartpos - cartwidth/2
        carttopy  = screen_height/2 - cartheight/2


        pygame.draw.rect(screen, (0,0,0), pygame.Rect(cartleftx, carttopy, cartwidth,cartheight))
        
        poleorigin = np.array([cartpos, carttopy])
        polelocalpoints = np.array([[-polewidth/2, 0],[polewidth/2,0],[polewidth/2,-poleheight],[-polewidth/2,-poleheight]])
        polerotmat = np.array([[np.cos(poleang),np.sin(poleang)],[-np.sin(poleang),np.cos(poleang)]])
        poleworldpoints = poleorigin[None,:] + polelocalpoints.dot(polerotmat)
        
        pygame.draw.polygon(screen, (0,255,0), poleworldpoints)
        
        if u:
            pygame.draw.line(screen, (255,0,0), (cartpos, screen_height/2), (cartpos + force*100, screen_height/2),5)
        pygame.display.flip()
