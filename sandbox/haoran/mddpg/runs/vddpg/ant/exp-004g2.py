"""
Variational DDPG (online, consevative)

U-shaped maze with Guassian reward
Compare to exp-004e and 004e2. Use pre-trained DDPG.
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
exp_prefix = "mddpg/vddpg/ant/" + exp_index
mode = "ec2"
subnet = "us-west-1b"
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
    def wall_offset(self):
        return [0, 0.5]

    # algo
    @variant
    def train_frequency(self):
        return [
            dict(
                actor_train_frequency=1,
                critic_train_frequency=1,
                update_target_frequency=1000,
                train_repeat=1,
            ),
        ]
    @variant
    def max_path_length(self):
        return [500]

    @variant
    def exp_info(self):
        return [
            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144921_250140_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=0
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144937_217560_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=100
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144939_034914_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=200
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144940_578952_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=300
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144941_933997_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=400
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144943_199058_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=500
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144944_759600_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=600
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144946_378760_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=700
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144947_821459_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=800
            ),

            dict(
                exp_prefix="mddpg/vddpg/ant/exp-004",
                exp_name="exp-004_20170218_144949_478466_tuomas_ant",
                snapshot_file="itr_499.pkl",
                env_name="tuomas_ant",
                seed=900
            ),
        ]

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
    shared_alg_kwargs = dict(
        max_path_length=v["max_path_length"],
        plt_backend=plt_backend,
        train_repeat=v["train_frequency"]["train_repeat"],
        actor_train_frequency=v["train_frequency"]["actor_train_frequency"],
        critic_train_frequency=v["train_frequency"]["critic_train_frequency"],
        update_target_frequency=v["train_frequency"]["update_target_frequency"],
        debug_mode=False,
        wall_offset=v["wall_offset"],
    )
    # algo
    if mode == "local_test" or mode == "local_docker_test":
        alg_kwargs = dict(
            epoch_length=200,
            min_pool_size=100,
                # beware that the algo doesn't finish an epoch
                # until it finishes one path
            n_epochs=100,
            eval_samples=1,
        )
    else:
        alg_kwargs = dict(
            epoch_length=10000,
            n_epochs=500,
            eval_samples=v["max_path_length"] * 10,
            min_pool_size=20000,
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


# new env
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from sandbox.haoran.mddpg.envs.mujoco.ant_puddle_env import AntPuddleEnv, Puddle
offset = {wall_offset}
puddles = [
    Puddle(x=-1, y=1 + offset, width=30, height=1, angle=0, cost=0,
        plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
    Puddle(x=-1, y=-2 - offset, width=30, height=1, angle=0, cost=0,
        plot_args=dict(color=(1., 0., 0., 1.0)), hard=True),
    Puddle(x=-2, y=-2 - offset, width=1, height=2 * (2 + offset),
        angle=0, cost=0, plot_args=dict(color=(1., 0., 0., 1.0)),
        hard=True),
]
env = TfEnv(normalize(
    AntPuddleEnv(
        reward_type=\"velocity\",
        direction=None,
        flip_thr=0.,
        puddles=puddles,
        mujoco_env_args=dict(
            random_init_state=False,
        ),
    ),
    clip=True,
))

from sandbox.haoran.mddpg.algos import ddpg
_algo = self.algo
qf_params = _algo.qf.get_param_values()
pi_params = _algo.policy.get_param_values()
# avoid errors from NeuralNetwork.get_copy()
ddpg.TARGET_PREFIX = "new_target_"
self.algo = ddpg.DDPG(
    env=env,
    exploration_strategy=_algo.exploration_strategy,
    policy=_algo.policy,
    qf=_algo.qf,
    eval_samples={eval_samples},
    plt_backend=\"{plt_backend}\",
    critic_train_frequency={critic_train_frequency},
    actor_train_frequency={actor_train_frequency},
    update_target_frequency={update_target_frequency},
    train_repeat={train_repeat},
    q_plot_settings=None,
    env_plot_settings=dict(),
    max_path_length={max_path_length},
    n_epochs={n_epochs},
    epoch_length={epoch_length},
    batch_size=_algo.batch_size,
    discount=_algo.discount,
    soft_target_tau=1,
    debug_mode={debug_mode},
)
self.algo.qf.set_param_values(qf_params)
self.algo.policy.set_param_values(pi_params)
self.algo.target_qf.set_param_values(qf_params)
self.algo.target_policy.set_param_values(pi_params)

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
