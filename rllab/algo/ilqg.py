from algo.optim import ilqg
from algo.optim.common import forward_pass, sample_actions
import numpy as np
from sampler.utils import rollout
from policy.linear_gaussian_policy import LinearGaussianPolicy

class ILQG(object):

    def __init__(self, linmode='shooting'):
        self.linmode = linmode
        pass

    def train(self, mdp):
        # plan directly on the states
        x0, _ = mdp.reset()
        ulb, uub = mdp.action_bounds
        uinit = sample_actions(ulb, uub, mdp.horizon)
        xinit = forward_pass(x0, uinit, f_forward=mdp.forward, f_cost=mdp.cost, f_final_cost=mdp.final_cost)["x"]
        mdp.plot(xinit, uinit, pause=True)
        for result in ilqg.solve(
                x0,
                uinit,
                f_forward=mdp.forward,
                f_cost=mdp.cost,
                f_final_cost=mdp.final_cost,
                grad_hints=mdp.grad_hints,
                ):
            print result["cost"]
            mdp.plot(result["x"], pause=True)
        #mdp.noise_level = 0.01
        #print 'rollout...'
        #rollout(mdp, LinearGaussianPolicy(u, K, k, x), max_length=mdp.horizon, animated=True, use_state=True)
        #while True:
        #    mdp.viewer.loop_once()
