from __future__ import print_function
from __future__ import absolute_import

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.snn.baselines.linear_feature_snn_baseline import LinearFeatureSNNBaseline
from sandbox.rocky.snn.bimod_env import BimodEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.snn.policies.stochastic_gaussian_mlp_policy import StochasticGaussianMLPPolicy
from sandbox.rocky.snn.algos.snn_algos import TRPO_snn
from sandbox.rocky.snn.hallucinators.prior_hallucinator import PriorHallucinator
from sandbox.rocky.snn.hallucinators.posterior_hallucinator import PosteriorHallucinator
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
import sys

stub(globals())

env = BimodEnv(mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False)

for seed in [11, 21, 31, 41, 51]:
    for n_samples in [5]:#0]:#[5]:#0, 5]:#xrange(1, 6):
        for self_normalize in [True, False]:
            policy = StochasticGaussianMLPPolicy(
                env_spec=env.spec,
                input_latent_vars=[('independent_bernoulli', 5)],
                hidden_latent_vars=[
                    [],
                    [],
                ],
                hidden_sizes=(8, 8)
            )

            baseline = LinearFeatureSNNBaseline(env_spec=env.spec)

            algo = TRPO_snn(
                env=env,
                policy=policy,
                baseline=baseline,
                self_normalize=self_normalize,
                # hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                hallucinator=PosteriorHallucinator(
                    env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples,
                    regressor_args=dict(use_trust_region=False),
                ),
                batch_size=500,
                whole_paths=True,
                max_path_length=100,
                n_itr=50,
                discount=0.99,
                step_size=0.01,
            )

            # for s in [4, 5, 155]:
            run_experiment_lite(
                stub_method_call=algo.train(),
                n_parallel=1,
                snapshot_mode="last",
                seed=seed,
                exp_prefix='snn_posterior',
            )

            sys.exit(0)
