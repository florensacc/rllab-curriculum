"""
Variational DDPG (online, consevative)

Retrain the policy and qf learned from exp-001 to move forward
"""
# imports -----------------------------------------------------
from sandbox.haoran.myscripts.retrainer import Retrainer

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
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220747_339282_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=0
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220748_166507_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=100
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220749_021102_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=200
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220749_736135_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=300
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220750_648493_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=400
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220751_484926_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=500
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220751_907818_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=600
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220752_303758_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=700
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220752_833379_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=800
                ),

                dict(
                    exp_prefix="tuomas/vddpg/exp-001",
                    exp_name="exp-001_20170212_220753_564335_tuomas_ant",
                    snapshot_file="itr_499.pkl",
                    env_name="tuomas_ant",
                    seed=900
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
            epoch_length=100,
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
self.algo.max_path_length={max_path_length}
self.algo.plt_backend=\"{plt_backend}\"
self.algo.n_epochs = {n_epochs}
self.algo.epoch_length = {epoch_length}
self.algo.min_pool_size = {min_pool_size}
self.algo.n_eval_samples = {eval_samples}
self.algo.env.set_param_values(
    dict(direction=(1., 0.))
)
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
