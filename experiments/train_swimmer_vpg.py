import os
os.environ['TENSORFUSE_MODE'] = 'theano'
os.environ['THEANO_FLAGS'] = 'device=cpu,optimizer=fast_compile'
from rllab.vf.mujoco_value_function import MujocoValueFunction
from rllab.mdp.swimmer_mdp import SwimmerMDP
from rllab.algo.vpg import VPG
from rllab.policy.mujoco_policy import MujocoPolicy
from functools import partial
import lasagne.updates

if __name__ == '__main__':
    mdp = SwimmerMDP()
    policy = MujocoPolicy(mdp, hidden_sizes=[32, 32])
    vf = MujocoValueFunction()
    algo = VPG(
            exp_name='swimmer_vpg',
            batch_size=500,
            max_path_length=500,
            plot=False,
            update_method=partial(lasagne.updates.adam, learning_rate=0.001),
    )
    algo.train(mdp, policy, vf)
