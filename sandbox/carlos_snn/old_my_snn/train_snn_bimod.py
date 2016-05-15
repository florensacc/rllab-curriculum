from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.carlos_snn.envs.bimod_env_tunable import BimodEnv
# from sandbox.carlos_snn.old_my_snn.npo_snn import NPO_snn
from sandbox.carlos_snn.old_my_snn.trpo_snn import TRPO_snn
from sandbox.carlos_snn.old_my_snn.s_mlp_policy import GaussianMLPPolicy_snn

from sandbox.carlos_snn.hallucinators.prior_hallucinator import PriorHallucinator

from rllab.envs.gym_env import GymEnv

stub(globals())

env = BimodEnv(mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False)
# env = BimodEnv(mu1=[1,0],mu2=[-1,0],sigma1=0.01,sigma2=0.01,rand_init=False)

# env = GymEnv('Hopper-v1', record_video=False)

for resample in [True, False]:
    for latent_dim in [0, 1, 2, 5]:
        for latent_type in ['bernoulli']:
            for n_samples in [0,1,4]:
                policy = GaussianMLPPolicy_snn(
                    env_spec=env.spec,
                    latent_dim=latent_dim,
                    latent_type=latent_type,
                    resample=resample,
                    hidden_sizes=(32, 32)
                )

                baseline = LinearFeatureBaseline(env_spec=env.spec)

                algo = TRPO_snn(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    # self_normalize=self_normalize,
                    hallucinator=PriorHallucinator(env_spec=env.spec, policy=policy, n_hallucinate_samples=n_samples),
                    batch_size=100,
                    whole_paths=True,
                    max_path_length=100,
                    n_itr=100,
                    discount=0.99,
                    step_size=0.01,
                )


                for s in [1,13,254]:
                    if resample:
                        exp_name = 'snn_{}_{}batch_{}latent_{}_{:04d}'.format('Resamp', 100, latent_dim, latent_type, s)
                    else:
                        exp_name = 'snn_{}_{}batch_{}latent_{}_{:04d}'.format('NoResamp', 100, latent_dim, latent_type, s)
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
                        exp_prefix='snn_test_bimod',
                        exp_name=exp_name,
                    )
