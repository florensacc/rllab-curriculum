"""
Image obs, image hash (bass)

Continue Frostbite in exp-026 for 200M frames.
Retrain takes place in a separate folder, so as to not screw up original data.
"""
# imports -----------------------------------------------------
from sandbox.haoran.myscripts.retrainer import Retrainer

""" others """
from sandbox.haoran.myscripts.myutilities import get_time_stamp
from sandbox.haoran.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import numpy as np
import sys,os
import copy

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup -----------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
exp_prefix = "bonus-trpo-atari/" + exp_index
mode = "ec2"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1b"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils

n_parallel = 2 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 400
plot = False
use_gpu = False
sync_s3_pkl = True
config.USE_TF = False

if "local" in mode and sys.platform == "darwin":
    set_cpu_affinity = False
    cpu_assignments = None
    serial_compile = False
else:
    set_cpu_affinity = True
    cpu_assignments = None
    serial_compile = True

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def exp_info(self):
        return [
            dict(
                exp_prefix="bonus-trpo-atari/exp-026",
                exp_name="exp-026_20161028_021829_837041_frostbite",
                game="frostbite",
                snapshot_file="itr_900.pkl",
                seed=0,
            ),
            # dict(
            #     exp_prefix="bonus-trpo-atari/exp-026",
            #     exp_name="exp-026_20161028_024039_131888_frostbite",
            #     game="frostbite",
            #     snapshot_file="itr_900.pkl",
            # ),
            dict(
                exp_prefix="bonus-trpo-atari/exp-026",
                exp_name="exp-026_20161028_024055_094577_frostbite",
                game="frostbite",
                snapshot_file="itr_900.pkl",
                seed=100,
            ),
            dict(
                exp_prefix="bonus-trpo-atari/exp-026",
                exp_name="exp-026_20161028_024053_593719_frostbite",
                game="frostbite",
                snapshot_file="itr_900.pkl",
                seed=200,
            ),
            dict(
                exp_prefix="bonus-trpo-atari/exp-026",
                exp_name="exp-026_20161028_024056_604919_frostbite",
                game="frostbite",
                snapshot_file="itr_900.pkl",
                seed=300,
            ),
            dict(
                exp_prefix="bonus-trpo-atari/exp-026",
                exp_name="exp-026_20161028_024059_774582_frostbite",
                game="frostbite",
                snapshot_file="itr_900.pkl",
                seed=400,
            ),
        ]

    @variant
    def target_itr(self):
        return [4000]

variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    use_parallel = True
    exp_info = v["exp_info"]
    game = exp_info["game"]
    seed = exp_info["seed"]

    configure_script = """
hash = self.algo.bonus_evaluator.hash
hash.counter = "tables"
import multiprocessing as mp
hash.tables_lock = mp.Value('i')
self.algo.sampler.avoid_duplicate_paths = False
self.algo.log_memory_usage = True
    """

    #-----------------------------------------------------

    exp_name = "{exp_index}_{time}_{game}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        game=game,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    if use_gpu:
        config.USE_GPU = True
        config.DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"

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
    retrainer = Retrainer(
        exp_prefix=exp_info["exp_prefix"],
        exp_name=exp_info["exp_name"],
        snapshot_file=exp_info["snapshot_file"],
        target_itr=v["target_itr"],
        n_parallel=n_parallel,
        configure_script=configure_script,
    )


    # launch -------------------------------------------
    if use_parallel:
        run_experiment_lite(
            retrainer.retrain(),
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            mode=actual_mode,
            variant=v,
            use_gpu=use_gpu,
            plot=plot,
            sync_s3_pkl=sync_s3_pkl,
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
            seed=seed,
        )
    else:
        raise NotImplementedError

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
