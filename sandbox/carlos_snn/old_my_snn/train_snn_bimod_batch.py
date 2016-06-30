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
        for reward_coef in [0.1, 1]:
            for latent_dim in [1, 2, 3, 4]:
                for latent_name in ['bernoulli']:
                    for n_samples in [0, 1, 2, 4]:
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

                        # set all the experiments that will run in the same in the same instance (just different seeds)
                        batch_tasks = []
                        for s in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                            if resample:
                                exp_name = 'snn_{}_{}rewcoef_{}latent_{}_{}hallu_{:04d}'.format(
                                    'Resamp', ''.join(str(reward_coef).split('.')), latent_dim, latent_name, n_samples,
                                    s)
                            else:
                                exp_name = 'snn_{}_{}rewcoef_{}latent_{}_{}hallu_{:04d}'.format(
                                    'NoResamp', ''.join(str(reward_coef).split('.')), latent_dim, latent_name,
                                    n_samples, s)
                            batch_tasks.append(dict(
                                stub_method_call=algo.train(),
                                exp_name=exp_name,
                                # Number of parallel workers for sampling
                                n_parallel=1,
                                # Only keep the snapshot parameters for the last iteration
                                snapshot_mode="last",
                                # Specifies the seed for the experiment. If this is not provided, a random seed
                                # will be used
                                seed=s,
                                # plot=True,
                            ))

                        run_experiment_lite(
                            batch_tasks=batch_tasks,
                            mode='ec2',  # remove plotting!!
                            # Save to data/local/exp_prefix/exp_name/ if running in mode="local"
                            exp_prefix='2snn_MI_10modes_1radius',
                        )
