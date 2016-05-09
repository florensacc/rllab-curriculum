from rllab.algos.npo import NPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from old_my_snn.npo_snn import NPO_snn
from old_my_snn.s_mlp_policy import GaussianMLPPolicy_snn
from sandbox.carlos_snn.bimod_env_tunable import BimodEnv

stub(globals())

env = BimodEnv(mu1=-1,mu2=1,sigma1=0.01,sigma2=0.01,rand_init=False)
# env = BimodEnv(mu1=[1,0],mu2=[-1,0],sigma1=0.01,sigma2=0.01,rand_init=False)

policy = GaussianMLPPolicy_snn(
    env_spec=env.spec,
    latent_dim=2,
    latent_type='normal',
    hidden_sizes=(8,8)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = NPO_snn(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=100,
    whole_paths=True,
    max_path_length=100,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
)
###the above is useless
for algorithm in [NPO]:
    for batch_size in [100,500,1000,2000]:
        # for latent_dim in [1,2,4,8,12]:
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                # latent_dim=latent_dim,
                # latent_type='normal',
                hidden_sizes=(8,8)
            )

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = algorithm(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=batch_size,
                whole_paths=True,
                max_path_length=100,
                n_itr=100,
                discount=0.99,
                step_size=0.01,
            )
            for seed in [4,5,155]:
                run_experiment_lite(
                    stub_method_call=algo.train(),
                    # Number of parallel workers for sampling
                    n_parallel=1,
                    # Only keep the snapshot parameters for the last iteration
                    snapshot_mode="last",
                    # Specifies the seed for the experiment. If this is not provided, a random seed
                    # will be used
                    seed=seed,
                    # plot=True,
                    # Save to data/local/exp_prefix/exp_name/
                    exp_prefix='npo-1Dbimod-batches',
                    exp_name='npo_{}batch_{:04d}'.format(batch_size,seed),
                )

# import plt_results1D
# plt_results1D.plot_all_exp("./data/local/"+'snn-1Dbimod-normal')