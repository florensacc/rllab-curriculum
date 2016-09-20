"""
Test runs on continuous tasks
Also compare with typical TRPO
"""
from sandbox.adam.parallel.trpo import ParallelTRPO
from sandbox.adam.parallel.linear_feature_baseline import ParallelLinearFeatureBaseline
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.ec2_info import instance_info, subnet_info
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv

from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import sys,os
import copy

# stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

exp_prefix = "parallel-trpo/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "local_test"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1a"

n_parallel = 4
snapshot_mode = "last"
store_paths = False
plot = False
use_gpu = False # should change conv_type and ~/.theanorc
sync_s3_pkl = True


# params ---------------------------------------
batch_size = 20000
max_path_length = 500
discount = 0.99
n_itr = 1000
cg_args = dict(
    cg_iters=10,
    subsample_factor=0.2,
) # worth tuning to see the speed up by parallelism


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0,100,200,300,400]

    @variant
    def env(self):
        return ["swimmer","hopper","walker","halfcheetah","ant","humanoid","cartpole","double_pendulum"]
    @variant
    def use_parallel(self):
        return [False,True]

variants = VG().variants()


print("#Experiments: %d" % len(variants))
for v in variants:
    env = v["env"]
    exp_name = "alex_{time}_{env}".format(
        time=get_time_stamp(),
        env=env,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    if env == "hopper":
        env = normalize(HopperEnv())
    elif env == "walker":
        env = normalize(Walker2DEnv())
    elif env == "halfcheetah":
        env = normalize(HalfCheetahEnv())
    elif env == "ant":
        env = normalize(AntEnv())
    elif env == "humanoid":
        env = normalize(SimpleHumanoidEnv)
    elif env == "cartpole":
        env = normalize(CartpoleEnv())
    elif env == "double_pendulum":
        env = normalize(DoublePendulumEnv)
    else:
        raise NotImplementedError

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )
    if v["use_parallel"]:
        baseline = ParallelLinearFeatureBaseline(env_spec=env.spec)

        algo = ParallelTRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            discount=discount,
            n_itr=n_itr,
            plot=plot,
            step_size=step_size,
            store_paths=store_paths,
            n_parallel=n_parallel,
            optimizer_args=cg_args,
        )
    else:
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=max_path_length,
            discount=discount,
            n_itr=n_itr,
            plot=plot,
            step_size=step_size,
            store_paths=store_paths,
            optimizer_args=cg_args,
        )


    # run --------------------------------------------------
    if "local_docker" in mode:
        actual_mode = "local_docker"
    elif "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
        # configure instance
        info = instance_info[ec2_instance]
        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"])
        n_parallel = int(info["vCPU"] /2)

        # choose subnet
        config.AWS_NETWORK_INTERFACES = [
            dict(
                SubnetId=subnet_info[subnet]["SubnetID"],
                Groups=subnet_info[subnet]["Groups"],
                DeviceIndex=0,
                AssociatePublicIpAddress=True,
            )
        ]
    else:
        raise NotImplementedError


    run_experiment_lite(
        algo.train(),
        script="sandbox/adam/run_experiment_lite_par.py",
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=v["seed"],
        snapshot_mode=snapshot_mode,
        mode=actual_mode,
        variant=v,
        use_gpu=use_gpu,
        plot=plot,
        sync_s3_pkl=sync_s3_pkl,
        terminate_machine=("test" not in mode),
        n_parallel=n_parallel,
    )

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
