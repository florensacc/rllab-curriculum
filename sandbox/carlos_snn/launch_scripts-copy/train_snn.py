from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.carlos_snn.hallucinators.prior_hallucinator import PriorHallucinator
from sandbox.carlos_snn.old_my_snn.s_mlp_policy import GaussianMLPPolicy_snn
from sandbox.carlos_snn.old_my_snn.trpo_snn import TRPO_snn
from sandbox.carlos_snn.regressors.latent_regressor import Latent_regressor

# from rllab.envs.gym_env import GymEnv  # for ec2 this doesn't work yet..
# from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
# from rllab.envs.mujoco.swimmer_env import SwimmerEnv
# from rllab.envs.mujoco.humanoid_env import HumanoidEnv

from rllab.envs.normalized_env import normalize

stub(globals())

# env = HalfCheetahEnv()  # GymEnv('HalfCheetah-v1', record_video=False)
env = normalize(SwimmerEnv())
# env = normalize(HumanoidEnv())
# env = normalize(SimpleHumanoidEnv())

for resample in [False]:
    if resample:
        recurrence_choice = [False]
    else:
        recurrence_choice = [True]
    recurrence_choice = [False]  ## JUST TO IMPOSE NON-RECURRENT
    for recurrent_reg in recurrence_choice:
        for reward_coef in [0]:
            for latent_dim in [0]:
                for latent_name in ['categorical']:
                    for n_samples in [0]:
                        for noisify_coef in [0.1]:
                            policy = GaussianMLPPolicy_snn(
                                env_spec=env.spec,
                                latent_dim=latent_dim,
                                latent_name=latent_name,
                                bilinear_integration=False,  # concatenate also the outer product
                                resample=resample,
                                hidden_sizes=(32, 32),  # (100, 50, 25),
                                min_std=1e-6,
                            )

                            baseline = LinearFeatureBaseline(env_spec=env.spec)

                            if latent_dim:
                                latent_regressor = Latent_regressor(
                                    env_spec=env.spec,
                                    policy=policy,
                                    recurrent=recurrent_reg,
                                    predict_all=True,  # use all the predictions and not only the last
                                    obs_regressed='all',  # [-3] is the x-position of the com, otherwise put 'all'
                                    act_regressed=[],  # use [] for nothing or 'all' for all.
                                    use_only_sign=False,  # for the regressor we use only the sign to estimate the post
                                    noisify_traj_coef=noisify_coef,
                                    optimizer=None,  # this defaults to LBFGS, for first order, put 'fist_order'
                                    regressor_args={
                                        'hidden_sizes': (32, 32),  # (100, 50, 25),
                                        'name': 'latent_reg',
                                        'use_trust_region': True,  # this is useless if using 'first_order'
                                    }
                                )
                            else:
                                latent_regressor = None

                            algo = TRPO_snn(
                                env=env,
                                policy=policy,
                                baseline=baseline,
                                self_normalize=True,
                                log_individual_latents=True,  # this will log the progress of every latent value!
                                log_deterministic=True,
                                logged_MI=[dict(recurrent=recurrent_reg,  #it will copy all but this (if other to copy,
                                                obs_regressed=[-3],         # code changes in npo_snn... to do)
                                                act_regressed=[],
                                                )
                                           ],  # for none use empty list [], for all use 'all_individual',
                                               # otherwise list of pairs, each entry a list of numbers ([obs],[acts])
                                                #### this sets a RECURRENT ONE!!
                                hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy,
                                                               n_hallucinate_samples=n_samples),
                                latent_regressor=latent_regressor,
                                reward_coef=reward_coef,
                                batch_size=50000,
                                whole_paths=True,
                                max_path_length=500,
                                n_itr=1000,
                                discount=0.99,
                                step_size=0.01,
                            )

                            for s in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                                if resample:
                                    exp_name = 'snn_{}UseA_{}noisy_{}_{}rewcoef_NoRecReg_{}latent_{}_{}hallu_NObilinear_{:04d}'.format(
                                        str(True), ''.join(str(noisify_coef).split('.')), 'Resamp',
                                        ''.join(str(reward_coef).split('.')), latent_dim, latent_name, n_samples, s)
                                else:
                                    if recurrent_reg:
                                        exp_name = 'snn_{}UseA_{}noisy_{}_{}rewcoef_RecReg_{}latent_{}_{}hallu_NObilinear_{:04d}'.format(
                                            str(True), ''.join(str(noisify_coef).split('.')), 'NoResamp',
                                            ''.join(str(reward_coef).split('.')), latent_dim, latent_name, n_samples, s)
                                    else:
                                        exp_name = 'snn_{}UseA_{}noisy_{}_{}rewcoef_NoRecReg_{}latent_{}_{}hallu_NObilinear_{:04d}'.format(
                                            str(True), ''.join(str(noisify_coef).split('.')), 'NoResamp',
                                            ''.join(str(reward_coef).split('.')), latent_dim, latent_name, n_samples, s)
                                run_experiment_lite(
                                    stub_method_call=algo.train(),
                                    mode='ec2',
                                    # Number of parallel workers for sampling
                                    n_parallel=8,
                                    # Only keep the snapshot parameters for the last iteration
                                    snapshot_mode="last",
                                    # Specifies the seed for the experiment. If this is not provided, a random seed
                                    # will be used
                                    seed=s,
                                    # plot=True,
                                    # Save to data/ec2/exp_prefix/exp_name/
                                    exp_prefix='snn_MI_humanoid-500pl-categorical-noisy-NoRecReg',
                                    exp_name=exp_name,
                                    sync_s3_pkl=True,  # for sync the pkl file also during the training
                                    terminate_machine=True,  # dangerous to have False!
                                )
