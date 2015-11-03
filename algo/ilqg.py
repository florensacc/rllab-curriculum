from .optim import ilqg
import numpy as np
from sampler.utils import rollout
from policy.linear_gaussian_policy import LinearGaussianPolicy

class ILQG(object):

    def __init__(self, linmode='shooting'):
        self.linmode = linmode
        pass

    def playback(self, x0, u, mdp):
        mdp.start_viewer()
        while True:
            print 'playing'
            s, _ = mdp.reset()
            for a in u:
                s, _, _, _ = mdp.step(s, a)
                mdp.viewer.loop_once()
                import time
                time.sleep(0.02)
            time.sleep(3)

    def train(self, mdp):
        # plan directly on the states
        x0, _ = mdp.reset()
        uinit = (np.random.rand(mdp.horizon, mdp.n_actions) - 0.5) * 0.1
        #du = K*dx
        u, K, k, x, Quu = ilqg.solve(
                x0,
                uinit,
                sysdyn=mdp.forward_dynamics,
                cost_func=mdp.cost,
                final_cost_func=mdp.final_cost,
        )
        import ipdb; ipdb.set_trace()
        mdp.noise_level = 0.01
        print 'rollout...'
        rollout(mdp, LinearGaussianPolicy(u, K, k, x), max_length=mdp.horizon, animated=True, use_state=True)
        while True:
            mdp.viewer.loop_once()
