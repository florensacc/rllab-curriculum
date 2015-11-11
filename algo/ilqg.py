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
        uinit = (np.random.rand(mdp.horizon, mdp.n_actions) - 0.5)*2
        xinit = ilqg.forward_pass(x0, uinit, f_forward=mdp.forward, f_cost=mdp.cost, f_final_cost=mdp.final_cost)["x"]
        mdp.plot(xinit, uinit, pause=True)
        for result in ilqg.solve(
                x0,
                uinit,
                f_forward=mdp.forward,
                f_cost=mdp.cost,
                f_final_cost=mdp.final_cost,
                grad_hints=mdp.grad_hints,
                ):
            mdp.plot(result["x"], pause=True)
        mdp.noise_level = 0.01
        print 'rollout...'
        rollout(mdp, LinearGaussianPolicy(u, K, k, x), max_length=mdp.horizon, animated=True, use_state=True)
        while True:
            mdp.viewer.loop_once()
