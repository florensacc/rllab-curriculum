from rllab.mdp.john_mjc.wrapper_mdp import WrapperMDP
import numpy as np


class SwimmerMDP(WrapperMDP):

    def __init__(self):
        super(SwimmerMDP, self).__init__("3swimmer")


if __name__ == "__main__":
    import time
    mdp = SwimmerMDP()
    state, obs = mdp.reset()
    action = [0, 1]
    while True:
        state = mdp.step(state, action)[0]
        mdp.plot()
        time.sleep(1.0/20)
