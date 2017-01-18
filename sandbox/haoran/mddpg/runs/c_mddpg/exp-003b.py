"""
Try TRPO on the double slit environment
"""
# imports -----------------------------------------------------
import tensorflow as tf
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import \
    GaussianMLPPolicy
from sandbox.rocky.tf.baselines.linear_feature_baseline import \
    LinearFeatureBaseline
from sandbox.haoran.mddpg.kernels.gaussian_kernel import \
    SimpleAdaptiveDiagonalGaussianKernel, \
    SimpleDiagonalConstructor, DiagonalGaussianKernel
from sandbox.haoran.myscripts.envs import EnvChooser
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler

""" others """
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import sys,os
import copy
import numpy as np

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup --------------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
exp_prefix = "mddpg/c_mddpg/" + exp_index
mode = "local"
ec2_instance = "c4.4xlarge"
subnet = "us-west-1c"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils

n_parallel = 1 # only for local exp
snapshot_mode = "last"
snapshot_gap = 100
plot = False
sync_s3_pkl = True

# variant params ---------------------------------------------------
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0,100,200]

    @variant
    def env_name(self):
        return [
            "double_slit",
            # "swimmer",
            # "hopper",
            # "walker",
            # "ant",
            # "halfcheetah",
            # "humanoid",
            # "cartpole",
            # "inv_double_pendulum",
        ]

    @variant
    def max_path_length(self):
        return [30]

variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params -----------------------------------
    # >>>>>>
    # algo
    seed=v["seed"]
    env_name = v["env_name"]
    optimizer_args = dict(
        cg_iters=100,
        reg_coeff=1e-3,
    )

    if mode == "local_test" or mode == "local_docker_test":
        algo_kwargs = dict(
            n_itr=5,
            batch_size=100,
            max_path_length=v["max_path_length"],
            force_batch_sampler=True,
            optimizer_args=optimizer_args,
        )
    else:
        algo_kwargs = dict(
            n_itr=100,
            batch_size=5000,
            max_path_length=v["max_path_length"],
            force_batch_sampler=True,
            optimizer_args=optimizer_args,
            step_size=0.1
        )
    if env_name == "hopper":
        env_kwargs = {
            "alive_coeff": 0.5
        }
    else:
        env_kwargs = {}

    # other exp setup --------------------------------------
    exp_name = "{exp_index}_{time}_{env_name}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        env_name=env_name
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

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
    elif "kube" in mode:
        actual_mode = "lab_kube"
        info = instance_info[ec2_instance]
        n_parallel = int(info["vCPU"] /2)

        config.KUBE_DEFAULT_RESOURCES = {
            "requests": {
                "cpu": int(info["vCPU"]*0.75)
            }
        }
        config.KUBE_DEFAULT_NODE_SELECTOR = {
            "aws/type": ec2_instance
        }
        exp_prefix = exp_prefix.replace('/','-') # otherwise kube rejects
    else:
        raise NotImplementedError

    # construct objects ----------------------------------
    env_chooser = EnvChooser()
    env = TfEnv(normalize(env_chooser.choose_env(env_name,**env_kwargs)))
    policy = GaussianMLPPolicy(
        "policy",
        env_spec=env.spec
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        **algo_kwargs
    )

    # run -----------------------------------------------------------

    run_experiment_lite(
        algo.train(),
        n_parallel=n_parallel,
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=seed,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        mode=actual_mode,
        variant=v,
        plot=plot,
        sync_s3_pkl=sync_s3_pkl,
        sync_log_on_termination=True,
        sync_all_data_node_to_s3=True,
    )

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
