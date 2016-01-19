# Source: Control-Limited Differential Dynamic Programming by Y Tassa et al.

from .base import ControlMDP
import numpy as np

class PendulumMDP(ControlMDP):

    def __init__(self):
        super(PendulumMDP, self).__init__()
        self.reset()

    def reset(self):
        self.state = np.array([np.pi, 0])
        self.t = 0
        return self.state, self.state

    def cost(self, state, action):
        R = 1e-5*np.eye(1)
        return 0.5 * action.T.dot(R).dot(action)

    def step(self, action):
        next_state = self.forward_dynamics(self.state, action)
        next_obs = next_state
        reward = - (next_state[0]**2)
        self.t += 1
        done = self.t == self.horizon
        if done:
            self.reset()
        return next_state, next_obs, reward, done

    def final_cost(self, state):
        xf = np.array([0, 0])
        Qf = np.diag([1, 1])
        return 0.5 * (state - xf).T.dot(Qf).dot(state - xf)

    @property
    def xinit(self):
        x0 = np.array([np.pi, 0])#, _ = self.reset()
        xf = np.array([0, 0])
        return np.array([np.linspace(s, e, self.horizon+1) for s, e in zip(x0, xf)]).T

    def forward_dynamics(self, state, action):
        g = 9.8
        m = 1.0 
        l = 1.0 # length of the rod
        mu = 0.01 # friction coefficient
        dt = 0.005
        th, thdot = state
        u0 = action[0]
        newth = th + thdot*dt
        newthdot = thdot + ( (-mu/(m*l**2))*thdot + (g/l)*np.sin(th) + u0/(m*l**2) ) * dt
        return np.array([newth, newthdot])

    @property
    def observation_shape(self):
        return (2,)

    @property
    def action_dim(self):
        return 1

    @property
    def observation_dtype(self):
        return 'float32'

    @property
    def action_dtype(self):
        return 'float32'


