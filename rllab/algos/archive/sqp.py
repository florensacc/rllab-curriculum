from algo.optim import sqp
from algo.optim.common import forward_pass, sample_actions
import numpy as np

class SQP(object):

    def __init__(self):
        pass

    def train(self, mdp):
        # plan directly on the states
        x0, _ = mdp.reset()
        ulb, uub = mdp.action_bounds
        uinit = sample_actions(ulb, uub, mdp.horizon)
        xinit = forward_pass(x0, uinit, f_forward=mdp.forward, f_cost=mdp.cost, f_final_cost=mdp.final_cost)["x"]
        mdp.plot(xinit, uinit, pause=True)
        for result in sqp.solve(
                x0,
                uinit,
                f_forward=mdp.forward,
                f_cost=mdp.cost,
                f_final_cost=mdp.final_cost,
                grad_hints=mdp.grad_hints,
                #state_bounds=mdp.state_bounds,
                #action_bounds=mdp.action_bounds,
                ):
            mdp.plot(result["x"], pause=True)
