# continue exp_028, but with parallel sampling for advantage computation

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
import git

# environments
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv

from rllab.envs.box2d.car_parking_env import CarParkingEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv

stub(globals())
os.path.join(config.PROJECT_PATH)

# check for uncommitted changes ------------------------------
repo = git.Repo('.')
if repo.is_dirty():
    answer = ''
    while answer not in ['y','Y','n','N']:
        answer = raw_input("The repository has uncommitted changes. Do you want to continue? (y/n)")
    if answer in ['n','N']:
        sys.exit(1)

# some important hyper-params that show up in the folder name -----------------
mode="ec2_parallel"
exp_prefix = os.path.basename(__file__).split('.')[0] # exp_xxx

# fixed params
algo_type = "trpo"
max_path_length=300
hidden_sizes = (32,32)
n_itr = 1000
foretrack_ratio = 1/0.95
max_foretracks = 100
backtrack_ratio = 0.95
max_backtracks = 100
cg_iters = 100
reg_coeff = 1e-3
batch_size = 10000
gae_lambda = 1.0
sf = 0.5
momentum_type = "v0"
momentum = 0

reuse_history=dict(
    max_len=1,
    reweigh_type="resample",
    reuse_for_gradient=True,
    reuse_for_kl=False,
)


# varying params
env_names = ["hopper"]
resample_list = [1]


# seeds
n_seed=5
seeds=np.arange(1,100*n_seed+1,100)

# mode-specific settings
if mode == "local_run":
    n_parallel = 4
    plot = True
elif mode == "local_test":
    n_parallel = 1
    plot = True
elif mode == "ec2":
    n_parallel = 1
    plot = False
elif mode == "ec2_parallel":
    n_parallel = 10
    config.AWS_INSTANCE_TYPE = "m4.10xlarge"
    config.AWS_SPOT_PRICE = '1.0'
    plot = False
else:
    raise NotImplementedError

# ------------------------------------------------------------------------------
exp_names = []

for env_name, prev_adv_resample_times in itertools.product(env_names,resample_list):
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
        elif env_name == "human":
            env = SimpleHumanoidEnv()
        else:
            raise NotImplementedError
        env = normalize(env)

        policy = GaussianMLPPolicy(
            init_std=1.0,
            env_spec=env.spec,
            hidden_sizes=hidden_sizes,
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
                gae_lambda=gae_lambda,
                plot=plot,
                store_paths=True,
                reuse_history=reuse_history,
                prev_adv_resample_times=prev_adv_resample_times,

                optimizer_args=dict(
                    cg_iters=cg_iters,
                    subsample_factor=sf,
                    foretrack_ratio=foretrack_ratio,
                    max_foretracks=max_foretracks,
                    backtrack_ratio=backtrack_ratio,
                    max_backtracks=max_backtracks,
                    reg_coeff=reg_coeff,
                    momentum=momentum,
                    momentum_type=momentum_type,
                )
            )

        import datetime
        import dateutil.tz
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y%m%d_%H%M%S')

        reuse = "h{max_len}{g}{kl}_{reweigh_type}_{resample_times}".format(
            max_len=reuse_history["max_len"],
            g="g" if reuse_history["reuse_for_gradient"] else "",
            kl="kl" if reuse_history["reuse_for_kl"] else "",
            reweigh_type=reuse_history["reweigh_type"],
            resample_times="%d"%(prev_adv_resample_times),
        )

        exp_name_prefix="alex_{time}_{env_name}_{reuse}_bs{batch_size}k".format(
            time=timestamp,
            env_name=env_name,
            reuse=reuse,
            batch_size=batch_size/1000,
        )
        exp_names.append("\"" + exp_name_prefix + "\",")

        for seed in seeds:
            exp_name = exp_name_prefix + "_s%d"%(seed)
            if "local" in mode:
                def run():
                    run_experiment_lite(
                        algo.train(),
                        exp_prefix=exp_prefix,
                        n_parallel=n_parallel,
                        snapshot_mode="all",
                        seed=seed,
                        plot=plot,
                        exp_name=exp_name,
                    )
            elif "ec2" in mode:
                if len(exp_name) > 64:
                    print "Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name)
                    sys.exit(1)
                def run():
                    run_experiment_lite(
                        algo.train(),
                        exp_prefix=exp_prefix,
                        n_parallel=n_parallel,
                        snapshot_mode="last",
                        seed=seed,
                        plot=plot,
                        exp_name=exp_name,

                        mode="ec2",
                        terminate_machine=True,
                    )
            else:
                raise NotImplementedError
            run()
            if "test" in mode:
                sys.exit(0)


# record the experiment names to a file
# also record the branch name and commit number
logs = []
logs += ["branch: %s" %(repo.active_branch.name)]
logs += ["commit SHA: %s"%(repo.head.object.hexsha)]
logs += exp_names

cur_script_name = __file__
log_file_name = cur_script_name.split('.py')[0] + '.log'
with open(log_file_name,'w') as f:
    for message in logs:
        f.write(message + "\n")

# make the current script read-only to avoid accidental changes after ec2 runs
if "local" not in mode:
    os.system("chmod 444 %s"%(__file__))
