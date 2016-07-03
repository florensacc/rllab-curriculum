from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.carlos_snn.envs.bimod_env_tunable import BimodEnv
from sandbox.carlos_snn.envs.multiMod2D_env import MultiModEnv
from sandbox.carlos_snn.old_my_snn.trpo_snn import TRPO_snn
# from sandbox.carlos_snn.old_my_snn.npo_snn_rewardMI import NPO_snn

from sandbox.carlos_snn.old_my_snn.s_mlp_policy import GaussianMLPPolicy_snn

from sandbox.carlos_snn.hallucinators.prior_hallucinator import PriorHallucinator

from sandbox.carlos_snn.regressors.latent_regressor import Latent_regressor

from rllab.envs.gym_env import GymEnv

stub(globals())

# env = BimodEnv(mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False)
# env = GymEnv('Hopper-v1', record_video=False)

for n in [2,3,4,5]:
    env=MultiModEnv(n=n)
    for resample in [True]:
        for reward_coef in [0]:
            for latent_dim in [1, 2, 5]:
                for latent_name in ['bernoulli','normal']:
                    for n_samples in [0,1,2]:
                        policy = GaussianMLPPolicy_snn(
                            env_spec=env.spec,
                            latent_dim=latent_dim,
                            latent_name=latent_name,
                            resample=resample,
                            hidden_sizes=(8, 8)  # remember to change size if using Gym!!!!!!
                        )

                        baseline = LinearFeatureBaseline(env_spec=env.spec)

                        # without latents
                        if latent_dim:
                            latent_regressor = Latent_regressor(
                                env_spec=env.spec,
                                policy=policy,
                                regressor_args={
                                    'hidden_sizes': (8, 8),
                                    'name': 'latent_reg'
                                }
                            )
                        else:
                            latent_regressor = None

                        algo = TRPO_snn(
                            env=env,
                            policy=policy,
                            baseline=baseline,
                            self_normalize=True,
                            hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                            latent_regressor=latent_regressor,
                            reward_coef=reward_coef,
                            batch_size=500,
                            whole_paths=True,
                            max_path_length=100,
                            n_itr=100,
                            discount=0.99,
                            step_size=0.01,
                        )

                        for s in [4,5,155]:
                            if resample:
                                exp_name = 'snn_{}_{}modes_{}rewcoef_{}batch_{}latent_{}_{}hallu_{:04d}'.format(
                                    'Resamp', n, reward_coef, 500, latent_dim, latent_name, n_samples, s)
                            else:
                                exp_name = 'snn_{}_{}modes_{}rewcoef_{}batch_{}latent_{}_{}hallu_{:04d}'.format(
                                    'NoResamp', n, reward_coef, 500, latent_dim, latent_name, n_samples, s)
                            run_experiment_lite(
                                stub_method_call=algo.train(),
                                mode='local',
                                # Number of parallel workers for sampling
                                n_parallel=1,
                                # Only keep the snapshot parameters for the last iteration
                                snapshot_mode="last",
                                # Specifies the seed for the experiment. If this is not provided, a random seed
                                # will be used
                                seed=s,
                                # plot=True,
                                # Save to data/local/exp_prefix/exp_name/
                                exp_prefix='snn_MI_2D',
                                exp_name=exp_name,
                            )
