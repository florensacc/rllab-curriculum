"""
Variational DDPG (online, consevative)

Re-train policies / qfs learned from exp-009b, b2 (VDDPG) but add forward
    reward to the env.
(need to use the VDDPG in commit a4005d5)
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
subnet = "us-west-1b"
ec2_instance = "c4.4xlarge"
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
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111634_998102_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111634_998665_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111634_999104_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111634_999503_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111634_999896_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111745_394798_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111745_395627_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111745_396189_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111745_396738_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111745_397246_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111747_555012_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111747_555865_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111747_556428_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111747_556962_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111747_557452_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111750_689025_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111750_689856_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111750_690368_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111750_690868_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111750_691372_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111753_011418_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111753_012262_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111753_012799_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111753_013327_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111753_013844_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111755_565188_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111755_566148_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111755_566706_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111755_567243_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111755_567753_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111757_678236_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111757_679076_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111757_679592_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111757_680111_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111757_680568_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111800_560201_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111800_561040_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111800_561606_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111800_562119_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111800_562623_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111802_938108_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111802_938966_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111802_939487_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111802_940001_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111802_940510_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111805_179853_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111805_180691_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111805_181207_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111805_181725_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111805_182229_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111809_740697_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111809_741533_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111809_742046_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111809_742553_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111809_743074_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111812_108567_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111812_109415_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111812_109932_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111812_110448_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/exp-009b2",
                exp_name="exp-009b2_20170126_111812_110910_gym_hopper",
                snapshot_file="itr_999.pkl",
                env_name="gym_hopper",
                seed=400
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
        )
    else:
        alg_kwargs = dict(
            epoch_length=1000,
            n_epochs=1000,
            n_eval_paths=10,
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
self.algo.max_path_length = {max_path_length}
self.algo.plt_backend = \"{plt_backend}\"
self.algo.n_epochs = {n_epochs}
self.algo.epoch_length = {epoch_length}
self.algo.min_pool_size = {min_pool_size}
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
