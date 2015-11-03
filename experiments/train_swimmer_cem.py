import os

os.environ['CGT_COMPAT_MODE'] = 'theano'
os.environ['THEANO_FLAGS'] = 'device=cpu'

from algo.cem import CEM
from mdp.swimmer_mdp import SwimmerMDP
from policy.mujoco_policy import MujocoPolicy

mdp = SwimmerMDP(horizon=500)
policy = MujocoPolicy(mdp)
CEM().train(mdp, policy)
