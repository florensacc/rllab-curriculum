import os
os.environ['CGT_COMPAT_MODE'] = 'tensorflow'
from mdp.cartpole_mdp import CartpoleMDP
from mdp.swimmer_mdp import SwimmerMDP

import numpy as np
np.random.seed(0)

mdp = SwimmerMDP()
state = mdp.reset()[0]
#state = np.random.uniform(low=-0.05*4., high=0.05*4., size=(4,))#mdp.reset()[0]
#print state
#mdp.plot()
#state = mdp.step(state, [0])[0]

for i in range(50):
    state = mdp.step(state, [2, 0])[0]
    #print state
    mdp.plot()
