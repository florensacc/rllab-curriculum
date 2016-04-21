from __future__ import print_function
from __future__ import absolute_import

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.snn.bimod_env import BimodEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.snn.policies.stochastic_gaussian_mlp_policy import StochasticGaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO

stub(globals())

env = BimodEnv(mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False)

policy = StochasticGaussianMLPPolicy(
    env_spec=env.spec,
    input_latent_vars=[('bernoulli', 2)],
    # latent_dim=2,
    # latent_type='binomial',
    hidden_sizes=(8, 8)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=500,
    whole_paths=True,
    max_path_length=100,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
)

# for s in [4, 5, 155]:
run_experiment_lite(
    stub_method_call=algo.train(),
    n_parallel=1,
    snapshot_mode="last",
    seed=4,
    exp_prefix='snn',
)
