import os
os.environ['TENSORFUSE_MODE'] = 'theano'
os.environ['THEANO_FLAGS'] = 'device=cpu'
import numpy as np
from rllab.policy.mujoco_policy import MujocoPolicy
#from rllab.mdp.mujoco.walker_mdp import WalkerMDP
from rllab.mdp.igor_mjc import AcrobotMDP
from rllab.sampler.utils import rollout

print 'reading data'
data = np.load('itr_350.npz')
print 'read data'

params = data['cur_policy_params']
print params.shape
mdp = AcrobotMDP()
policy = MujocoPolicy(mdp, hidden_sizes=[32, 32])#30,30])
print policy.get_param_values().shape
policy.set_param_values(params)
# zero out the variance
#policy.log_std_vars[0].set_value(np.ones_like(policy.log_std_vars[0].get_value()) * -100)
#cur_params = policy.get_param_values()
result = rollout(mdp, policy, max_length=50, animated=False)#mdp.demo_policy(policy)
import ipdb; ipdb.set_trace()


