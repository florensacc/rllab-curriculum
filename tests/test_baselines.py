import os

os.environ['THEANO_FLAGS'] = 'mode=FAST_COMPILE,optimizer=None'

from rllab.algos.vpg import VPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc import ext
from nose2 import tools


algo = VPG(n_itr=1, batch_size=1000, max_path_length=100)

baselines = [ZeroBaseline, LinearFeatureBaseline, GaussianMLPBaseline]

@tools.params(*baselines)
def test_baseline(baseline_cls):
    env = CartpoleEnv()
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(6,))
    baseline = baseline_cls(env_spec=env.spec)
    algo.train(env=env, policy=policy, baseline=baseline)
