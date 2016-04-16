
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.carlos_snn.bimod_env_tunable import BimodEnv
from rllab.envs.normalized_env import normalize
from sandbox.carlos_snn.s_mlp_policy import GaussianMLPPolicy_snn
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.carlos_snn.npo_snn import NPO_snn

stub(globals())

env = BimodEnv(mu1=-1,mu2=1,sigma1=0.01,sigma2=0.01,rand_init=False)
# env = BimodEnv(mu1=[1,0],mu2=[-1,0],sigma1=0.01,sigma2=0.01,rand_init=False)

policy = GaussianMLPPolicy_snn(
    env_spec=env.spec,
    hidden_sizes=(8,8)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = NPO_snn(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=400,
    whole_paths=True,
    max_path_length=100,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
)


run_experiment_lite(
    stub_method_call=algo.train(),
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="all",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
    # Save to data/local/exp_name_timestamp
    exp_prefix='snn_ppo_try',
)
