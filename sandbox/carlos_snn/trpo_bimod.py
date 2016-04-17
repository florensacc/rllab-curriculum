from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite

# from sandbox.carlos_snn.bimod_env_tunable import BimodEnv
from sandbox.carlos_snn.bimod2D_env import BimodEnv

stub(globals())

# env = BimodEnv(mu1=-1,mu2=1,sigma1=0.01,sigma2=0.01,rand_init=True)
env = BimodEnv(mu1=[1,0],mu2=[-1,0],sigma1=0.01,sigma2=0.01,rand_init=False)

## If you want to recover a previous policy
# import joblib
# import numpy as np
# datafile = joblib.load('./data/local/' +
#  'trpo-2Dbimod-baseline200-just-check/trpo_2Dbimod_baseline200_just_check_2016_04_16_14_51_55_0001/' +
#  'itr_157.pkl')
# policy = datafile['policy']
# print np.any(np.isnan(policy.get_param_values()))
# print policy

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(8,8)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=500,
    whole_paths=True, ##I think this is useless: the kw only appears in some provisional/experimental files by Rocky or PChen
    store_paths=True,
    max_path_length=100,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
)

for s in [4]:#,5,155]:
    run_experiment_lite(
        stub_method_call=algo.train(),
        # Number of parallel workers for sampling
        n_parallel=1,  
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=s,
        # plot=True,
        # Save to data/local/exp_name/exp_name_timestamp ##OJO! the folder exp_name will change _ by -!!
        exp_prefix='trpo_2Dbimod_baseline200_just_check',
    )
