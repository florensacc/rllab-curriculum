from rllab.mdp.john_mjc2.wrapper_mdp import WrapperMDP
import numpy as np


class Walker2DMDP(WrapperMDP):

    def __init__(self):
        super(Walker2DMDP, self).__init__("walker2d")


if __name__ == "__main__":
    import time
    mdp = Walker2DMDP()
    state, obs = mdp.reset()
    #action = np.random.rand(mdp.action_dim)#[0, 1, 0, 0, 0, 0]
    while True:
        action = np.random.rand(mdp.action_dim)*180#[0, 1, 0, 0, 0, 0]
        state = mdp.step(state, action)[0]
        mdp.plot()
        time.sleep(1.0/20)
