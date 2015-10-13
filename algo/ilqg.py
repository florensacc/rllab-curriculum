from .optim import ilqg
import numpy as np

class ILQG(object):

    def __init__(self, linmode='shooting'):
        self.linmode = linmode
        pass

    def playback(self, x0, u, mdp):
        #mdp.
        mdp.start_viewer()
        while True:
            print 'playing'
            s, _ = mdp.reset()
            for a in u:
                s, _, _, _ = mdp.step(s, a)#, autoreset=False)
                mdp.viewer.loop_once()
                import time
                time.sleep(0.02)
            time.sleep(3)

    def train(self, gen_mdp):
        mdp = gen_mdp()
        # plan directly on the states
        x0, _ = mdp.reset()
        uinit = np.zeros((mdp.horizon - 1, mdp.n_actions))
        u, _, _ = ilqg.solve(
                x0,
                uinit,
                sysdyn=mdp.forward_dynamics,
                cost_func=mdp.cost,
                final_cost_func=mdp.final_cost,
        )
        mdp.demo(u)
