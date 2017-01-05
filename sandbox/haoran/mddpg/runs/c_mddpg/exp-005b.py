"""
Conservative version of MDDPG

Try the MultiGoalEnv
"""
# imports -----------------------------------------------------
import tensorflow as tf
from sandbox.haoran.mddpg.algos.mddpg import MDDPG
from sandbox.haoran.mddpg.policies.mnn_policy import \
    FeedForwardMultiPolicy, MNNStrategy
from sandbox.haoran.mddpg.kernels.gaussian_kernel import \
    SimpleAdaptiveDiagonalGaussianKernel, \
    SimpleDiagonalConstructor, DiagonalGaussianKernel
from sandbox.haoran.mddpg.qfunctions.nn_qfunction import FeedForwardCritic
from sandbox.haoran.myscripts.envs import EnvChooser
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
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
mode = "local_test"
ec2_instance = "c4.4xlarge"
subnet = "us-west-1b"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils

n_parallel = 1 # only for local exp
snapshot_mode = "all"
snapshot_gap = 10
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
            "multi_goal"
        ]
    @variant
    def K(self):
        return [2, 4, 8]

    @variant
    def alpha(self):
        return [1, 10, 100]

    @variant
    def sigma(self):
        return [1e-3]

    @variant
    def max_path_length(self):
        return [15]

    @variant
    def policy_learning_rate(self):
        return [1e-4]

    @variant
    def theta(self):
        return [0]

    @variant
    def qf_extra_training(self):
        return [0]

    @variant
    def switch_type(self):
        return [
            # "per_action",
            "per_path"
        ]

    @variant
    def q_target_type(self):
        return [
            "mean",
            "max"
        ]

variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params -----------------------------------
    # >>>>>>
    # algo
    seed=v["seed"]
    env_name = v["env_name"]
    K = v["K"]
    adaptive_kernel = False
    sigma = v["sigma"]
    theta = v["theta"]

    shared_ddpg_kwargs = dict(
        alpha=v["alpha"],
        max_path_length=v["max_path_length"],
        policy_learning_rate=v["policy_learning_rate"],
        qf_extra_training=v["qf_extra_training"],
        q_target_type = v["q_target_type"],
    )
    if mode == "local_test" or mode == "local_docker_test":
        ddpg_kwargs = dict(
            epoch_length = 1000,
            min_pool_size = 2,
            eval_samples = 100,
            n_epochs=50,
            batch_size=4,
        )
    else:
        ddpg_kwargs = dict(
            epoch_length=10000,
            batch_size=64,
            n_epochs=100,
        )
    ddpg_kwargs.update(shared_ddpg_kwargs)
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

    es = MNNStrategy(
        K=K,
        substrategy=OUStrategy(env_spec=env.spec,theta=theta),
        switch_type=v["switch_type"],
    )
    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        embedded_hidden_sizes=(8,8),
        observation_hidden_sizes=(8,),
    )
    policy = FeedForwardMultiPolicy(
        "actor",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        K=K,
    )
    if K > 1 and adaptive_kernel:
        kernel = SimpleAdaptiveDiagonalGaussianKernel(
            "kernel",
            dim=env.action_space.flat_dim,
        )
    else:
        diag_constructor = SimpleDiagonalConstructor(
            dim=env.action_space.flat_dim,
            sigma=sigma,
        )
        kernel = DiagonalGaussianKernel(
            "kernel",
            diag=diag_constructor.diag(),
        )
    algorithm = MDDPG(
        env=env,
        exploration_strategy=es,
        policy=policy,
        kernel=kernel,
        qf=qf,
        K=K,
        **ddpg_kwargs
    )

    # run -----------------------------------------------------------

    run_experiment_lite(
        algorithm.train(),
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
