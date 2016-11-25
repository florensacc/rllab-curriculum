"""
Further acceleration
- parallel evaluation
- reduced snapshot frequency
"""
# imports -----------------------------------------------------
import tensorflow as tf
from sandbox.haoran.mddpg.algos.ddpg import DDPG
from sandbox.haoran.mddpg.policies.nn_policy import FeedForwardPolicy
from sandbox.haoran.mddpg.qfunctions.nn_qfunction import FeedForwardCritic
from sandbox.haoran.myscripts.envs import EnvChooser
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler

""" others """
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import sys,os
import copy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup --------------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
exp_prefix = "mddpg/tests/" + exp_index
mode = "ec2_test"
ec2_instance = "c4.2xlarge"
subnet = "us-west-1c"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils

n_parallel = 2 # only for local exp
snapshot_mode = "last"
snapshot_gap = 200
plot = False
sync_s3_pkl = True

# variant params ---------------------------------------------------
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0,100,200,300,400]

    @variant
    def env_name(self):
        return [
            "halfcheetah",
            # "swimmer","hopper","halfcheetah","ant","humanoid",
            # "cartpole","double_pendulum","inv_double_pendulum",
        ]

variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params -----------------------------------
    # >>>>>>
    # algo
    seed=v["seed"]
    env_name = v["env_name"]

    if mode == "local_test":
        ddpg_kwargs = dict(
            epoch_length = 100,
            min_pool_size = 100,
            eval_samples = 100,
            eval_epoch_gap=10,
        )
    else:
        ddpg_kwargs = dict(
            eval_epoch_gap=25,
        )

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
    env = TfEnv(env_chooser.choose_env(env_name))

    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
    )
    policy = FeedForwardPolicy(
        "actor",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
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
