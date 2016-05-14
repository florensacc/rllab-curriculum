from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.carlos_snn.envs.bimod_env_tunable import BimodEnv
from sandbox.carlos_snn.old_my_snn.npo_snn import NPO_snn
from sandbox.carlos_snn.old_my_snn.s_mlp_policy import GaussianMLPPolicy_snn

stub(globals())

env = BimodEnv(mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False)
# env = BimodEnv(mu1=[1,0],mu2=[-1,0],sigma1=0.01,sigma2=0.01,rand_init=False)

policy = GaussianMLPPolicy_snn(
    env_spec=env.spec,
    latent_dim=2,
    latent_type='binomial',
    hidden_sizes=(8,8)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = NPO_snn(
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


for s in [4]:
    run_experiment_lite(
        stub_method_call=algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=s,
        # plot=True,
        # Save to data/local/exp_prefix/exp_name/
        exp_prefix='snn_test',
        exp_name='snn_npo_{}batch_{}latent_bino4_{:04d}'.format(500,2,s),
    )
