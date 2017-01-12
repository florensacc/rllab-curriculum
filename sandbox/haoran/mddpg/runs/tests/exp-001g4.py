"""
Test speed with multiple instances. New docker image.
Try MDDPG with different K and kernel adaptation.
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

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup --------------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
exp_prefix = "mddpg/tests/" + exp_index
mode = "ec2"
subnet = "us-west-1c"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3:latest" # needs psutils
config.AWS_IMAGE_ID = "ami-85d181e5" # with docker already pulled

n_parallel = 1 # only for local exp
snapshot_mode = "last"
snapshot_gap = 200
plot = False
sync_s3_pkl = True

# variant params ---------------------------------------------------
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0]

    @variant
    def env_name(self):
        return [
            # "swimmer",
            "hopper",
            # "walker",
            # "ant",
            # "halfcheetah",
            # "humanoid",
            # "cartpole",
            # "inv_double_pendulum",
        ]
    @variant
    def batch_size(self):
        return [
            64
        ]
    @variant
    def alive_coeff(self):
        return [0.5]

    @variant
    def ec2_instance(self):
        return [
            "c4.large",
            "c4.xlarge",
            "c4.2xlarge",
            "c4.4xlarge",
        ]
    @variant
    def adaptive_kernel(self):
        return [True, False]
    @variant
    def alpha(self):
        return [0.1]
    @variant
    def K(self):
        return [1, 8, 16]

variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params -----------------------------------
    # >>>>>>
    # algo
    seed=v["seed"]
    env_name = v["env_name"]
    ec2_instance = v["ec2_instance"]
    adaptive_kernel = v["adaptive_kernel"]
    K = v["K"]

    if mode == "local_test" or mode == "local_docker_test":
        ddpg_kwargs = dict(
            alpha = v["alpha"],
            epoch_length = 1000,
            min_pool_size = 100,
            eval_samples = 100,
        )
    else:
        ddpg_kwargs = dict(
            alpha = v["alpha"],
            epoch_length=2000,
            min_pool_size=1000,
            batch_size=v["batch_size"],
            n_epochs=10,
        )
    env_kwargs = {
        "alive_coeff": v["alive_coeff"]
    }

    # other exp setup --------------------------------------
    exp_name = "{exp_index}_{time}_{env_name}_{instance}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        env_name=env_name,
        instance=ec2_instance.replace('.','_')
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
        env_chooser.choose_env(env_name,**env_kwargs)
    ))

    qf = FeedForwardCritic(
        "critic",
        env.observation_space.flat_dim,
        env.action_space.flat_dim,
    )
    substrategy = OUStrategy(
        env_spec=env.spec,
    )
    es = MNNStrategy(
        K=K,
        substrategy=substrategy,
        switch_type="per_path",
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
