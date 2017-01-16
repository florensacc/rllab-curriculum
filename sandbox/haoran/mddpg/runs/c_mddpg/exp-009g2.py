"""
Conservative version of MDDPG

Continue exp-009e, f. With extra hyper-param sweep. Hopefully we can increase
the chance of observing swimming in two directions.
* q_target_type
* policy_learning_rate
* scale_reward
* exploration noise
* alpha (redundant)
- switch_type = "per_action"

Change settings to speedup training
* network size: try 100 (Vitchyr's) or 32 (Carlos', TRPO default)
    (btw, the original DDPG uses (400, 300) units)
- batch_size: I'm not sure how small we can make it
"""

# imports -----------------------------------------------------
import tensorflow as tf
import joblib
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
from sandbox.haoran.mddpg.gaussian_strategy import GaussianStrategy
from sandbox.haoran.mddpg.qfunctions.interpolate_qfunction \
    import InterpolateQFunction, DataLoader
from sandbox.haoran.mddpg.misc.annealer import LinearAnnealer


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
mode = "ec2"
ec2_instance = "c4.2xlarge"
subnet = "us-west-1c"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3:latest" # needs psutils
config.AWS_IMAGE_ID = "ami-85d181e5" # with docker already pulled

n_task_per_instance = 5
n_parallel = 2 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 50
plot = False

# variant params ---------------------------------------------------
class VG(VariantGenerator):
    @variant
    def zzseed(self):
        return [0,100,200,300,400]

    @variant
    def env_name(self):
        return [
            "swimmer_undirected"
        ]
    @variant
    def K(self):
        return [8, 16]

    @variant
    def alpha(self):
        return [1.]

    @variant
    def scale_reward(self):
        return [0.01, 0.1, 1., 10., 100.]
        # for policy update, scaling up reward is like decreasing the
        # temperature; I choose update to 100 because good alpha seems between
        # 0.01 and 0.1.

    @variant
    def policy_learning_rate(self):
        return [1e-5, 1e-4, 1e-3, 1e-2]

    @variant
    def max_path_length(self):
        return [500]

    @variant
    def switch_type(self):
        return ["per_action"]

    @variant
    def q_target_type(self):
        return ["max"]

    @variant
    def n_units(self):
        return [32, 100]

variants = VG().variants()
batch_tasks = []
print("#Experiments: %d" % len(variants))
for v in variants:
    if v["switch_type"] == "per_path" and v["q_target_type"] == "max":
        continue

    # non-variant params -----------------------------------
    # >>>>>>
    # algo
    seed=v["zzseed"]
    env_name = v["env_name"]
    K = v["K"]
    adaptive_kernel = True
    units = v["n_units"]

    shared_ddpg_kwargs = dict(
        alpha=v["alpha"],
        max_path_length=v["max_path_length"],
        eval_max_head_repeat=1,
        batch_size=64,
        q_target_type=v["q_target_type"],
        policy_learning_rate=v["policy_learning_rate"],
        scale_reward=v["scale_reward"],
    )
    if mode == "local_test" or mode == "local_docker_test":
        ddpg_kwargs = dict(
            n_epochs=5,
            epoch_length=10,
            min_pool_size=100,
        )
    else:
        ddpg_kwargs = dict(
            n_epochs=500,
            epoch_length=2000,
        )
    ddpg_kwargs.update(shared_ddpg_kwargs)
    exp_name = "{exp_index}_{time}_{env_name}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        env_name=env_name,
    )
    if env_name == "hopper":
        env_kwargs = {
            "alive_coeff": 0.5
        }
    elif env_name == "swimmer_undirected":
        plot_variant = copy.deepcopy(v)
        plot_variant["exp_name"] = exp_name
        env_kwargs = {
            "visitation_plot_config": {
                "mesh_density": 50,
                "prefix": '',
                "variant": plot_variant,
            }
        }
    else:
        env_kwargs = {}

    # other exp setup --------------------------------------
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
    env = TfEnv(normalize(
        env_chooser.choose_env(env_name,**env_kwargs)
    ))

    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        observation_hidden_sizes=(units,),
        embedded_hidden_sizes=(units,),
    )
    substrategy = OUStrategy(
        env_spec=env.spec,
    )
    es = MNNStrategy(
        K=K,
        substrategy=substrategy,
        switch_type=v["switch_type"],
    )
    policy = FeedForwardMultiPolicy(
        "actor",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        K=K,
        shared_hidden_sizes=(units,),
        independent_hidden_sizes=(units,),
    )
    if K > 1 and adaptive_kernel:
        kernel = SimpleAdaptiveDiagonalGaussianKernel(
            "kernel",
            dim=env.action_space.flat_dim,
        )
    else:
        diag_constructor = SimpleDiagonalConstructor(
            dim=env.action_space.flat_dim,
            sigma=0.01,
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
    print(v)
    batch_tasks.append(
        dict(
            stub_method_call=algorithm.train(),
            exp_name=exp_name,
            seed=seed,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            variant=v,
            plot=plot,
            n_parallel=n_parallel,
        )
    )
    if len(batch_tasks) >= n_task_per_instance:
        run_experiment_lite(
            batch_tasks=batch_tasks,
            exp_prefix=exp_prefix,
            mode=actual_mode,
            sync_s3_pkl=True,
            sync_s3_log=True,
            sync_s3_png=True,
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
            terminate_machine=True,
        )
        batch_tasks = []
        if "test" in mode:
            sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
