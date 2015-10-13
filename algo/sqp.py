from .optim import sqp
import numpy as np

class SQP(object):

    def __init__(self):
        pass

    def train(self, gen_mdp):
        mdp = gen_mdp()
        x0, _ = mdp.reset()
        uinit = np.zeros((mdp.horizon - 1, mdp.n_actions))
        u = sqp.solve(
                x0,
                uinit,
                sysdyn=mdp.forward_dynamics,
                cost_func=mdp.cost,
                final_cost_func=mdp.final_cost,
        )
