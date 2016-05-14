from __future__ import print_function
from __future__ import absolute_import

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from sandbox.rocky.snn.baselines.linear_feature_snn_baseline import LinearFeatureSNNBaseline
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv

# from rllab.algos.trpo import TRPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.envs.normalized_env import normalize

# from sandbox.carlos_snn.snn_algos_test import TRPO_snn_test
# from sandbox.carlos_snn.hallucinators.prior_hallucinator import PriorHallucinator
from sandbox.carlos_snn.hallu_algos import TRPO_hallu
from sandbox.carlos_snn.hallu_algos import VPG_hallu

from sandbox.rocky.snn.policies.stochastic_gaussian_mlp_policy import StochasticGaussianMLPPolicy
# from sandbox.rocky.snn.algos.snn_algos import TRPO_snn
# from sandbox.rocky.snn.hallucinators.prior_hallucinator import PriorHallucinator
# from sandbox.rocky.snn.hallucinators.posterior_hallucinator import PosteriorHallucinator
from rllab.misc.instrument import stub, run_experiment_lite

# import sys

stub(globals())



# env = BimodEnv(mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False)
# env = normalize(CartpoleEnv())
env = CartpoleSwingupEnv()

# baseline1 = [LinearFeatureSNNBaseline(env_spec=env.spec),'SNNbaseline']
baseline2 = [LinearFeatureBaseline(env_spec=env.spec),'baseline']

for base in [baseline2]:
    # for latent_dim in [0]:#[0,1,2,5,11]:
        for n_samples in [0,1,2,10]:
            # for hid_latent in [0]:
            #     if latent_dim == 0:
            #         latents_def = None
            #     else:
            #         latents_def = [('independent_bernoulli', latent_dim)]
            #     policy_snn = StochasticGaussianMLPPolicy(
            #         env_spec=env.spec,
            #         input_latent_vars=latents_def,
            #         hidden_latent_vars=[
            #         #     # [('independent_gaussian', hid_latent)],
            #         #     # [('independent_gaussian', hid_latent)],
            #             [],
            #             [],
            #         ],
            #         hidden_sizes=(32, 32)
            #     )

                policy_gau = GaussianMLPPolicy(
                    env_spec=env.spec,
                    hidden_sizes=(32,32)
                )

                baseline = base[0]

                # algo = TRPO_snn_test(
                #     env=env,
                #     policy=policy_gau,
                #     baseline=baseline,
                #     self_normalize=False,
                #     hallucinator = None,
                #     n_samples=n_samples,
                #     # hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy_snn, n_hallucinate_samples=n_samples),
                #     # hallucinator=PosteriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                #     batch_size=1000,
                #     whole_paths=True,
                #     max_path_length=100,
                #     n_itr=10,
                #     discount=0.99,
                #     step_size=0.01,
                # )

                algo_hallu = TRPO_hallu(
                    env=env,
                    policy=policy_gau,
                    baseline=baseline,
                    self_normalize=True,
                    # hallucinator=None,
                    n_samples=n_samples,
                    # n_samples=n_samples,
                    # hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy_snn, n_hallucinate_samples=n_samples),
                    # hallucinator=PosteriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                    batch_size=2000,
                    whole_paths=True,
                    max_path_length=100,
                    n_itr=50,
                    discount=0.99,
                    step_size=0.01,
                )

                for s in [4]:
                    run_experiment_lite(
                        stub_method_call=algo_hallu.train(),
                        n_parallel=1,
                        snapshot_mode="last",
                        seed=s,
                        exp_prefix='halluciante_no_latents_test_hallu',
                        exp_name='cartSwing-notNormalized_trpo_{}Hallu_{:04d}'.format(
                             n_samples, s),
                    )
