"""
Variational DDPG (online, consevative)

Re-train policies / qfs learned from exp-009 (DDPG) but add forward
    reward to the env.
"""
# imports -----------------------------------------------------
from sandbox.haoran.myscripts.retrainer import Retrainer
# import tensorflow as tf
# import joblib
# from rllab.envs.normalized_env import normalize
# from rllab.exploration_strategies.ou_strategy import OUStrategy
# from sandbox.rocky.tf.envs.base import TfEnv
# from sandbox.haoran.myscripts.envs import EnvChooser
# from sandbox.tuomas.mddpg.kernels.gaussian_kernel import \
#     SimpleAdaptiveDiagonalGaussianKernel
# from sandbox.tuomas.mddpg.critics.nn_qfunction import FeedForwardCritic
# from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy
# from sandbox.tuomas.mddpg.policies.stochastic_policy import \
#     DummyExplorationStrategy, StochasticPolicyMaximizer
# from sandbox.tuomas.mddpg.algos.vddpg import VDDPG

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
subnet = "us-west-1c"
ec2_instance = "c4.2xlarge"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils
config.AWS_IMAGE_ID = "ami-85d181e5" # with docker already pulled

n_task_per_instance = 1
n_parallel = 4 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 10
plot = False

# variant params ---------------------------------------------------
class VG(VariantGenerator):
    @variant
    def exp_info(self):
        return [
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214229_352375_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=0, # not necessarily correspond to the original seeds
                scale_reward=1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214242_010439_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=400, # not necessarily correspond to the original seeds
                scale_reward=1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214244_905507_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=300, # not necessarily correspond to the original seeds
                scale_reward=0.1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214247_098873_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=200, # not necessarily correspond to the original seeds
                scale_reward=10,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214239_892841_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=100, # not necessarily correspond to the original seeds
                scale_reward=1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214242_848034_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=0, # not necessarily correspond to the original seeds
                scale_reward=0.1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214245_374721_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=400, # not necessarily correspond to the original seeds
                scale_reward=0.1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214247_574551_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=300, # not necessarily correspond to the original seeds
                scale_reward=10,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214240_779506_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=200, # not necessarily correspond to the original seeds
                scale_reward=1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214243_361339_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=100, # not necessarily correspond to the original seeds
                scale_reward=0.1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214245_876973_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=0, # not necessarily correspond to the original seeds
                scale_reward=10,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214248_281384_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=400, # not necessarily correspond to the original seeds
                scale_reward=10,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214241_275547_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=300, # not necessarily correspond to the original seeds
                scale_reward=1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214244_116509_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=200, # not necessarily correspond to the original seeds
                scale_reward=0.1,
            ),
            dict(
                exp_prefix="mddpg/vddpg/exp-009",
                exp_name="exp-009_20170125_214246_609084_gym_hopper",
                snapshot_file="itr_499.pkl",
                env_name="gym_hopper",
                seed=100, # not necessarily correspond to the original seeds
                scale_reward=10,
            ),
        ]
    @variant
    def max_path_length(self):
        return [500]

variants = VG().variants()
batch_tasks = []
print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params -----------------------------------
    # >>>>>>
    if "local" in mode and sys.platform == "darwin":
        plt_backend = "MacOSX"
    else:
        plt_backend = "Agg"
    # algo
    if mode == "local_test" or mode == "local_docker_test":
        alg_kwargs = dict(
            epoch_length=10,
            min_pool_size=20,
                # beware that the algo doesn't finish an epoch
                # until it finishes one path
            n_epochs=1,
            eval_samples=100,
        )
    else:
        alg_kwargs = dict(
            epoch_length=1000,
            n_epochs=1000,
            eval_samples=v["max_path_length"] * 10,
            min_pool_size=10000,
        )
    shared_alg_kwargs = dict(
        scale_reward=v["exp_info"]["scale_reward"],
        max_path_length=v["max_path_length"],
        plt_backend=plt_backend,
    )
    alg_kwargs.update(shared_alg_kwargs)

    # other exp setup --------------------------------------
    exp_name = "{exp_index}_{time}_{env_name}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        env_name=v["exp_info"]["env_name"],
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
    exp_info = v["exp_info"]

    # the script should have not indentation
    configure_script = """
qf_params = self.snapshot["qf"].get_param_values()
pi_params = self.snapshot["policy"].get_param_values()
from sandbox.haoran.mddpg.algos.ddpg import DDPG
self.algo = DDPG(
    env=self.snapshot["env"],
    policy=self.snapshot["policy"],
    exploration_strategy=self.snapshot["es"],
    qf=self.snapshot["qf"],
    scale_reward={scale_reward},
    max_path_length={max_path_length},
    plt_backend=\"{plt_backend}\",
)
self.algo.qf.set_param_values(qf_params)
self.algo.policy.set_param_values(pi_params)
self.algo.n_epochs = {n_epochs}
self.algo.epoch_length = {epoch_length}
self.algo.min_pool_size = {min_pool_size}
self.algo.n_eval_samples = {eval_samples}
self.algo.env.use_forward_reward = True
    """.format(**alg_kwargs)

    retrainer = Retrainer(
        exp_prefix=exp_info["exp_prefix"],
        exp_name=exp_info["exp_name"],
        snapshot_file=exp_info["snapshot_file"],
        configure_script=configure_script,
    )

    # run -----------------------------------------------------------
    print(v)
    batch_tasks.append(
        dict(
            stub_method_call=retrainer.retrain(),
            exp_name=exp_name,
            seed=v["exp_info"]["seed"],
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
        )
        batch_tasks = []
        if "test" in mode:
            sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
