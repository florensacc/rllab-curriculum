import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
os.environ['THEANO_FLAGS'] = 'device=cpu'
from mdp.swimmer_mdp import SwimmerMDP
from mdp.swimmer_mdp_bk import SwimmerMDP as SwimmerMDPBk
from mdp.cartpole_mdp import CartpoleMDP
import numpy as np
import timeit

mdp = CartpoleMDP()
state = mdp.reset()[0]
print timeit.timeit('mdp.step(state, np.array([0]))[0]', 'from __main__ import ' + ', '.join(globals()), number=1000)
#mdp_bk = SwimmerMDPBk()
#state = mdp_bk.reset()[0]
###
###print state
#
##for i in range(100): #import timeit
#print timeit.timeit('mdp_bk.step(state, np.array([1, 0]))[0]', 'from __main__ import ' + ', '.join(globals()), number=1000)
#
##    state =     #print state
#mdp = SwimmerMDP()
#state = mdp.reset()[0]
##state = mdp.step(state, np.array([1, 0]))[0]
##print state
#print timeit.timeit('mdp.step(state, np.array([1, 0]))[0]', 'from __main__ import ' + ', '.join(globals()), number=1000)
##for i in range(100):
##    state = mdp.step(state, [2, 0])[0]
##    print state
