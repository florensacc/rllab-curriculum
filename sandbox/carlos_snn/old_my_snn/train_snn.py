from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.carlos_snn.old_my_snn.trpo_snn import TRPO_snn
import sys

from sandbox.carlos_snn.old_my_snn.s_mlp_policy import GaussianMLPPolicy_snn

from sandbox.carlos_snn.hallucinators.prior_hallucinator import PriorHallucinator

from sandbox.carlos_snn.regressors.mlp_latent_regressor import MLPLatent_regressor

# from rllab.envs.gym_env import GymEnv  # for ec2 this doesn't work yet..
# from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from sandbox.carlos_snn.envs.half_cheetah_env import HalfCheetahEnv #change the reward to abs()
from rllab.envs.normalized_env import normalize

stub(globals())

env = HalfCheetahEnv() #GymEnv('HalfCheetah-v1', record_video=False)

for resample in [False]:
    if resample:
        recurrence_choice = [False]
    else:
        recurrence_choice = [True]
    for recurrent_reg in recurrence_choice:
        for reward_coef in [0.1]:
            for latent_dim in [1]:
                for latent_name in ['bernoulli']:
                    for n_samples in [0]:
                        policy = GaussianMLPPolicy_snn(
                            env_spec=env.spec,
                            latent_dim=latent_dim,
                            latent_name=latent_name,
                            resample=resample,
                            hidden_sizes=(32, 32)  # remember to change size if using Gym!!!!!!
                        )

                        baseline = LinearFeatureBaseline(env_spec=env.spec)

                        if latent_dim:
                            latent_regressor = MLPLatent_regressor(
                                env_spec=env.spec,
                                policy=policy,
                                recurrent=recurrent_reg,
                                regressor_args={
                                    'hidden_sizes': (32, 32),
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
                            log_individual_latents=True, #this will log the progress of every latent value!
                            hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy,
                                                           n_hallucinate_samples=n_samples   ),
                            latent_regressor=latent_regressor,
                            reward_coef=reward_coef,
                            batch_size=8000,
                            whole_paths=True,
                            max_path_length=500,
                            n_itr=500,
                            discount=0.99,
                            step_size=0.01,
                        )

                        for s in [5]: #[4, 5, 155, 234, 333]:
                            if resample:
                                exp_name = 'snn_{}_{}rewcoef_NoRecReg_{}latent_{}_{}hallu_{:04d}'.format(
                                    'Resamp', reward_coef, latent_dim, latent_name, n_samples, s)
                            else:
                                if recurrent_reg:
                                    exp_name = 'snn_{}_{}rewcoef_RecReg_{}latent_{}_{}hallu_{:04d}'.format(
                                        'NoResamp', reward_coef, latent_dim, latent_name, n_samples, s)
                                else:
                                    exp_name = 'snn_{}_{}rewcoef_NoRecReg_{}latent_{}_{}hallu_{:04d}'.format(
                                        'NoResamp', reward_coef, latent_dim, latent_name, n_samples, s)
                            run_experiment_lite(
                                stub_method_call=algo.train(),
                                mode='ec2',
                                # Number of parallel workers for sampling
                                n_parallel=4,
                                # Only keep the snapshot parameters for the last iteration
                                snapshot_mode="last",
                                # Specifies the seed for the experiment. If this is not provided, a random seed
                                # will be used
                                seed=s,
                                # plot=True,
                                # Save to data/local/exp_prefix/exp_name/
                                exp_prefix='snn_cheetah_try',
                                exp_name=exp_name,
                                terminate_machine=True, #dangerous to have False!
                            )
