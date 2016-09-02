# 20160615_235800
# trpo-momentum
# try larger networks for ant and see how momentum and cg_iters help

from rllab.algos.trpo import TRPO
from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
import lasagne.nonlinearities as NL
import lasagne
import sys
import numpy as np
import os
import itertools

from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv

from rllab.envs.box2d.car_parking_env import CarParkingEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv

stub(globals())
os.path.join(config.PROJECT_PATH)


# some important hyper-params that show up in the folder name -----------------
mode="ec2"
algo_type = "trpo"
max_path_length=500
n_itr=500
exp_prefix="trpo_momentum"

batch_sizes = np.array([20]) * 1000
env_names = ["ant"]
cg_iters_list = [10,30,100,500]

n_seed=10
seeds=np.arange(1,100*n_seed+1,100)

if mode == "local":
    n_parallel = 1
elif mode == "ec2":
    n_parallel = 1
elif mode == "ec2_parallel":
    n_parallel = 10
    config.AWS_INSTANCE_TYPE = "m4.10xlarge"
    config.AWS_SPOT_PRICE = '1.0'
else:
    raise NotImplementedError

# ------------------------------------------------------------------------------
exp_names = []

for batch_size,env_name,cg_iters,cg_init_from_prev in \
    itertools.product(batch_sizes,env_names,cg_iters_list,[True,False]):
        assert(np.mod(batch_size,1000)==0)
        if env_name == "swimmer":
            env = SwimmerEnv()
        elif env_name == "hopper":
            env = HopperEnv()
        elif env_name == "halfcheetah":
            env = HalfCheetahEnv()
        elif env_name == "walker":
            env = Walker2DEnv()
        elif env_name == "car_parking":
            env = CarParkingEnv()
        elif env_name == "cartpole_swingup":
            env = CartpoleSwingupEnv()
        elif env_name == "mountain_car": 
            env = MountainCarEnv()
        elif env_name in ["double_pendulum","dpend"]:
            env = DoublePendulumEnv()
        elif env_name == "cartpole":
            env = CartpoleEnv()
        elif env_name == "ant":
            env = AntEnv()
        else: 
            raise NotImplementedError
        env = normalize(env)

        policy = GaussianMLPPolicy(
            init_std=1.0,
            env_spec=env.spec,
            hidden_sizes=(64,64),
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)
        # baseline = ZeroBaseline(env_spec=env.spec)


        if algo_type == "trpo":
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=batch_size,
                max_path_length=max_path_length,
                n_itr=n_itr,
                discount=0.99,
                step_size=0.01,
                plot=False,
                store_paths=True,

                optimizer_args=dict(
                    cg_iters=cg_iters,
                    cg_init_from_prev=cg_init_from_prev,
                )
            )

        import datetime
        import dateutil.tz
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y%m%d_%H%M%S')

        
        # so hard to find a short name
        exp_name_prefix="alex_{time}_{env_name}_cgi{cg_iters}{init}_bs{batch_size}k".format(
            time=timestamp,
            env_name=env_name,
            cg_iters=cg_iters,
            init="_R" if cg_init_from_prev else "",
            batch_size=batch_size/1000,
        )
        exp_names.append("\"" + exp_name_prefix + "\",")

        for seed in seeds:
            exp_name = exp_name_prefix + "_s%d"%(seed)
            if mode=="local":
                def run():
                    run_experiment_lite(
                        algo.train(),
                        exp_prefix=exp_prefix,
                        n_parallel=n_parallel,
                        snapshot_mode="all",
                        seed=seed,
                        plot=True,
                        exp_name=exp_name,
                    )
            elif (mode=="ec2") or (mode=="ec2_parallel"):
                if len(exp_name) > 64:
                    print "Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name)
                    sys.exit(1)
                def run():
                    run_experiment_lite(
                        algo.train(),
                        exp_prefix=exp_prefix,
                        n_parallel=n_parallel, 
                        snapshot_mode="all",
                        seed=seed,
                        plot=False,
                        exp_name=exp_name,

                        mode="ec2",
                        terminate_machine=True,
                    )
            else:
                raise NotImplementedError
            run()


# record the experiment names to a file
cur_script_name = __file__
log_file_name = cur_script_name.split('.py')[0] + '.log'
with open(log_file_name,'w') as f:
    for exp_name in exp_names:
        f.write(exp_name + "\n")

