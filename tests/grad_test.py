import os
os.environ['TENSORFUSE_MODE'] = 'theano'
from mdp.swimmer_mdp import SwimmerMDP
import numpy as np

mdp = SwimmerMDP()
state = mdp.reset()[0]
#np.random.seed(0)
#state = np.random.rand(4)
action = np.array([1, 1])

from algo.optim.ilqg import linearize

result = linearize(np.array([state, state]), np.array([action]), mdp.forward, mdp.cost, mdp.final_cost)
print 'df_dx'
print mdp.grad_hints['df_dx'](state, action)
print result['fx'][0]
print 'df_du'
print mdp.grad_hints['df_du'](state, action)
print result['fu'][0]
print 'dc_du'
print mdp.grad_hints['dc_du'](state, action)
print result['cu'][0]
print 'dc_dx'
print mdp.grad_hints['dc_dx'](state, action)
print result['cx'][0]
print 'dc_duu'
print mdp.grad_hints['dc_duu'](state, action)
print result['cuu'][0]

print 'dc_dxu'
print mdp.grad_hints['dc_dxu'](state, action)
print result['cxu'][0]
