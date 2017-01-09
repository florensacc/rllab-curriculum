"""
Conservative version of MDDPG

Try SVGD with the pre-computed Q^*. Try switching a head per time step.
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
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.haoran.mddpg.qfunctions.interpolate_qfunction \
    import InterpolateQFunction, DataLoader


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
subnet = "us-west-1b"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils
config.AWS_IMAGE_ID = "ami-309ccd50" # with docker already pulled

n_task_per_instance = 10
n_parallel = 4 # only for local exp
snapshot_mode = "all"
snapshot_gap = 10
plot = False
sync_s3_pkl = True

# variant params ---------------------------------------------------
class VG(VariantGenerator):
    @variant
    def zzseed(self):
        return [0,100,200,300,400,500,600,700,800,900]

    @variant
    def env_name(self):
        return [
            "multi_goal"
        ]
    @variant
    def K(self):
        return [4, 8, 16]

    @variant
    def alpha(self):
        return [0, 0.01, 0.1, 1]

    @variant
    def sigma(self):
        return [1e-2]

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
            "per_action",
            # "per_path"
        ]

    @variant
    def q_target_type(self):
        return [
            # "mean",
            "max"
        ]

variants = VG().variants()
batch_tasks = []
print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params -----------------------------------
    # >>>>>>
    # algo
    seed=v["zzseed"]
    env_name = v["env_name"]
    K = v["K"]
    adaptive_kernel = True
    sigma = v["sigma"]
    theta = v["theta"]

    shared_ddpg_kwargs = dict(
        alpha=v["alpha"],
        max_path_length=v["max_path_length"],
        policy_learning_rate=v["policy_learning_rate"],
        qf_extra_training=v["qf_extra_training"],
        q_target_type = v["q_target_type"],
        only_train_actor=True,
    )
    if mode == "local_test" or mode == "local_docker_test":
        ddpg_kwargs = dict(
            epoch_length = 1000,
            min_pool_size = 2,
            eval_samples = 100,
            n_epochs=50,
            batch_size=64,
        )
    else:
        ddpg_kwargs = dict(
            epoch_length=1000,
            batch_size=64,
            n_epochs=50,
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

    data_file = "sandbox/haoran/mddpg/envs/multi_goal_goal_4_grid_0.5.pkl"
    if "ec2" in mode:
        data_file = "/root/code/rllab/" + data_file
        # need to use absolute path, since the python command is not executed
        # in the rllab directory
    data_loader = DataLoader(data_file, 'mQ')
    grid_size = 0.5
    s_grid_sizes = [grid_size] * 2
    a_grid_sizes = s_grid_sizes
    qf = InterpolateQFunction(
        scope_name="qf",
        discrete_Q=data_loader.load(),
        env_spec=env.spec,
        s_grid_sizes=s_grid_sizes,
        a_grid_sizes=a_grid_sizes,
    )

    es = MNNStrategy(
        K=K,
        substrategy=OUStrategy(env_spec=env.spec,theta=theta),
        switch_type=v["switch_type"],
    )
    policy = FeedForwardMultiPolicy(
        "actor",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        K=K,
        shared_hidden_sizes=tuple(),
        independent_hidden_sizes=(8,8),
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
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
            terminate_machine="test" not in mode,
        )
        batch_tasks = []
        if "test" in mode:
            sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
