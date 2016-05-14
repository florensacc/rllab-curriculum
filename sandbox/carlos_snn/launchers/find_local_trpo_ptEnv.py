from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.carlos_snn.envs.multiMod2D_env import MultiModEnv
from sandbox.carlos_snn.envs.bimod_env_tunable import BimodEnv

import datetime
import dateutil.tz
now = datetime.datetime.now(dateutil.tz.tzlocal())


stub(globals())

# env = BimodEnv(mu1=-1,mu2=1,sigma1=0.01,sigma2=0.01,rand_init=False)
# env = MultiModEnv(mu=[1,0], sigma=0.01,n=5,rand_init=False)

## If you want to recover a previous policy
# import joblib
# import numpy as np
# datafile = joblib.load('./data/local/' +
#  'trpo-2Dbimod-baseline200-just-check/trpo_2Dbimod_baseline200_just_check_2016_04_16_14_51_55_0001/' +
#  'itr_157.pkl')
# policy = datafile['policy']
# print np.any(np.isnan(policy.get_param_values()))
# print policy


for eps in [0.01]: # eps makes mu1 module higher.
    for disp in [0.5, 1]:  # this is how much the modes are shifted to the left
        env = BimodEnv(eps=eps, disp=disp, mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False)

        policy = GaussianMLPPolicy(  # this is not SNN!!
            env_spec=env.spec,
            hidden_sizes=(8, 8)
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=500,
            whole_paths=True,
            ##I think this is useless: the kw only appears in some provisional/experimental files by Rocky or PChen
            store_paths=True,
            max_path_length=100,
            n_itr=100,
            discount=0.99,
            step_size=0.01,
        )

        for s in [4,5,155]:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
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
                # Save to data/local/exp_name/exp_name_timestamp ##OJO! the folder exp_name will change _ by -!!
                exp_prefix='trpo-local-opt2',
                exp_name='trpo-local-opt2_eps{}_disp{}_s{:04d}_{} '.format(
                    ''.join(str(eps).split('.')), ''.join(str(disp).split('.')), s, timestamp),
            )
