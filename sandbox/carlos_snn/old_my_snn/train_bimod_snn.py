from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.carlos_snn.envs.multiMod2D_env import MultiModEnv
from sandbox.carlos_snn.old_my_snn.trpo_snn import TRPO_snn
import sys

from sandbox.carlos_snn.old_my_snn.s_mlp_policy import GaussianMLPPolicy_snn

from sandbox.carlos_snn.hallucinators.prior_hallucinator import PriorHallucinator

from sandbox.carlos_snn.regressors.mlp_latent_regressor import MLPLatent_regressor

stub(globals())

env = MultiModEnv(mu=(1, 0), sigma=0.01, n=10, rand_init=False)
# env = HalfCheetahEnv()#GymEnv('HalfCheetah-v1', record_video=False)

for resample in [False]:
    if resample:
        recurrence_choice = [False]
    else:
        recurrence_choice = [True]
    for recurrent_reg in recurrence_choice:
        for reward_coef in [0]:  # , 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5]: #, 0.1, 1]:
            for latent_dim in [4]:  # , 1, 2, 3, 4, 5]:
                for latent_name in ['bernoulli']:
                    for n_samples in [0, 2]:  # , 1, 2, 4, 8]:
                        policy = GaussianMLPPolicy_snn(
                            env_spec=env.spec,
                            min_std=1e-4,
                            latent_dim=latent_dim,
                            latent_name=latent_name,
                            resample=resample,
                            hidden_sizes=(8, 8)  # remember to change size if using Gym!!!!!!
                        )

                        baseline = LinearFeatureBaseline(env_spec=env.spec)

                        # without latents
                        if latent_dim:
                            latent_regressor = MLPLatent_regressor(
                                env_spec=env.spec,
                                policy=policy,
                                recurrent=True,
                                regressor_args={
                                    'hidden_sizes': (8, 8),
                                    'name': 'latent_reg',
                                }
                            )
                        else:
                            latent_regressor = None

                        algo = TRPO_snn(
                            env=env,
                            policy=policy,
                            baseline=baseline,
                            self_normalize=True,
                            hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy,
                                                           n_hallucinate_samples=n_samples),
                            latent_regressor=latent_regressor,
                            reward_coef=reward_coef,
                            batch_size=500,
                            whole_paths=True,
                            max_path_length=100,
                            n_itr=300,
                            discount=0.99,
                            step_size=0.01,
                        )

                        for s in [10, 20, 30, 40, 50]:  # , 5, 155, 234, 333]:
                            if resample:
                                exp_name = 'snn_{}_{}rewcoef_{}latent_{}_{}hallu_{:04d}'.format(
                                    'Resamp', reward_coef, latent_dim, latent_name, n_samples, s)
                            else:
                                exp_name = 'snn_{}_{}rewcoef_{}latent_{}_{}hallu_{:04d}'.format(
                                    'NoResamp', reward_coef, latent_dim, latent_name, n_samples, s)
                            run_experiment_lite(
                                stub_method_call=algo.train(),  # now it has plotting at the same time.
                                mode='local',
                                # Number of parallel workers for sampling
                                n_parallel=4,
                                # Only keep the snapshot parameters for the last iteration
                                snapshot_mode="last",
                                # Specifies the seed for the experiment. If this is not provided, a random seed
                                # will be used
                                seed=s,
                                # plot=True,
                                # Save to data/local/exp_prefix/exp_name/
                                exp_prefix='snn_MI_10modes_1radius',
                                exp_name=exp_name,
                            )
