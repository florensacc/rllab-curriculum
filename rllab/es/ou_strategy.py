from rllab.es.base import ExplorationStrategy
from rllab.misc.overrides import overrides
from rllab.misc.ext import AttrDict
from rllab.misc import autoargs
from rllab.core.serializable import Serializable
import numpy as np
import numpy.random as nr


class OUStrategy(ExplorationStrategy, Serializable):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.

    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    @autoargs.arg('mu', type=float,
                  help='Mean parameter for the Ornstein-Uhlenbeck process')
    @autoargs.arg('theta', type=float,
                  help='Feedback scaling parameter for the Ornstein-Uhlenbeck '
                       'process')
    @autoargs.arg('sigma', type=float,
                  help='Standard deviation term for the Ornstein-Uhlenbeck '
                       'process')
    def __init__(self, mdp, mu=0, theta=0.15, sigma=0.3, **kwargs):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = mdp.action_dim
        self.state = np.ones(self.action_dim) * self.mu
        self.action_bounds = mdp.action_bounds
        self.episode_reset()
        Serializable.__init__(self, mdp, mu, theta, sigma, **kwargs)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["state"] = self.state
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.state = d["state"]

    @overrides
    def episode_reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, *self.action_bounds)


if __name__ == "__main__":
    ou = OUStrategy(mdp=AttrDict(action_dim=1), mu=0, theta=0.15, sigma=0.3)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state()[0])
    import matplotlib.pyplot as plt
    plt.plot(states)
    plt.show()
