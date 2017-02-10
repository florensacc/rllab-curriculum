"""
Variational DDPG (online, consevative)

Compare the speed of GPU and CPU (for VDDPG).
Adapted from exp-009b2.
"""
# imports -----------------------------------------------------
import tensorflow as tf
import joblib
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.haoran.myscripts.envs import EnvChooser
from sandbox.tuomas.mddpg.kernels.gaussian_kernel import \
    SimpleAdaptiveDiagonalGaussianKernel
from sandbox.tuomas.mddpg.critics.nn_qfunction import FeedForwardCritic
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy
from sandbox.tuomas.mddpg.policies.stochastic_policy import \
    DummyExplorationStrategy, StochasticPolicyMaximizer
from sandbox.tuomas.mddpg.algos.vddpg import VDDPG

from sandbox.haoran.mddpg.algos.ddpg import DDPG
from sandbox.haoran.mddpg.policies.nn_policy import FeedForwardPolicy

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
exp_prefix = "mddpg/vddpg/" + exp_index
mode = "ec2"
subnet = "us-west-2b"
ec2_instance = "p2.xlarge"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3-tf-gpu" # needs psutils
config.AWS_IMAGE_ID = "ami-bc0cb5dc" # with docker already pulled
config.AWS_REGION_NAME = "us-west-2"
config.AWS_KEY_NAME = "hrtang-us-west-2-new"
config.TF_USE_GPU = True

n_task_per_instance = 1
n_parallel = 2 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 10
plot = False

# variant params ---------------------------------------------------
class VG(VariantGenerator):
    @variant
    def zzseed(self):
        return [0]

    @variant
    def env_name(self):
        return [
            "gym_hopper"
        ]
    @variant
    def max_path_length(self):
        return [500]

    @variant
    def ou_sigma(self):
        return [0.3]

    @variant
    def scale_reward(self):
        return [1]

    @variant
    def qf_learning_rate(self):
        return [1e-3]

    @variant
    def tau(self):
        return [1e-2]

    @variant
    def use_forward_reward(self):
        return [False]


variants = VG().variants()
batch_tasks = []
print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params -----------------------------------
    # >>>>>>
    # algo
    seed=v["zzseed"]
    env_name = v["env_name"]
    shared_ddpg_kwargs = dict(
        max_path_length=v["max_path_length"],
        scale_reward=v["scale_reward"],
        qf_learning_rate=v["qf_learning_rate"],
        soft_target_tau=v["tau"],
    )
    if "local" in mode and "local_docker" not in mode:
        shared_ddpg_kwargs["plt_backend"] = "MacOSX"
    else:
        shared_ddpg_kwargs["plt_backend"] = "Agg"

    if mode == "local_test" or mode == "local_docker_test":
        ddpg_kwargs = dict(
            epoch_length = 100,
            min_pool_size = 100,
            eval_samples = 100,
        )
    else:
        ddpg_kwargs = dict(
            epoch_length=1000,
            n_epochs=500,
            eval_samples=v["max_path_length"] * 10,
                # deterministic env and policy: only need 1 traj sample
        )
    ddpg_kwargs.update(shared_ddpg_kwargs)
    if env_name == "hopper":
        env_kwargs = {
            "alive_coeff": 0.5
        }
    elif env_name in ["swimmer_undirected", "tuomas_hopper"]:
        env_kwargs = {
            "random_init_state": False
        }
    elif env_name == "gym_hopper":
        env_kwargs = {
            "use_forward_reward": v["use_forward_reward"]
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
    env = TfEnv(normalize(
        env_chooser.choose_env(env_name,**env_kwargs),
        clip=True,
    ))
    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
        observation_hidden_sizes=(),
        embedded_hidden_sizes=(100, 100),
    )
    es = OUStrategy(
        env_spec=env.spec,
        mu=0,
        theta=0.15,
        sigma=v["ou_sigma"],
        clip=True,
    )
    policy = FeedForwardPolicy(
        scope_name="actor",
        observation_dim=env.observation_space.flat_dim,
        action_dim=env.action_space.flat_dim,
        output_nonlinearity=tf.nn.tanh,
        observation_hidden_sizes=(100, 100),
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
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
            terminate_machine=("test" not in mode),
            use_gpu=config.TF_USE_GPU,
        )
        batch_tasks = []
        if "test" in mode:
            sys.exit(0)
if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
