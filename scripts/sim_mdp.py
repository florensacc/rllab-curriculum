import os
os.environ['TENSORFUSE_MODE'] = 'theano'
from mdp.cartpole_mdp import CartpoleMDP
#from mdp.swimmer_mdp import SwimmerMDP

import numpy as np
np.random.seed(0)

mdp = CartpoleMDP()
state = mdp.reset()[0]
#state = np.random.uniform(low=-0.05*4., high=0.05*4., size=(4,))#mdp.reset()[0]
#print state
#mdp.plot()
#state = mdp.step(state, [0])[0]

import time
#start = time.time()
for i in range(1000):
    state = mdp.step(state, [0])[0]
    mdp.plot()
#print time.time() - start
    #print state
