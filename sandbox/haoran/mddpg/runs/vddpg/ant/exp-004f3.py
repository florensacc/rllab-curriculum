"""
Variational DDPG (online, consevative)

Train a DDPG counterpart of 004e
An exact comparison to 004e7
"""
# imports -----------------------------------------------------
import tensorflow as tf
from sandbox.haoran.mddpg.envs.mujoco.ant_puddle_env import \
    AntPuddleGenerator
from sandbox.haoran.mddpg.algos.ddpg import DDPG
from sandbox.haoran.mddpg.policies.nn_policy import FeedForwardPolicy
from sandbox.haoran.mddpg.qfunctions.nn_qfunction import FeedForwardCritic
from sandbox.haoran.myscripts.envs import EnvChooser
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
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
exp_prefix = "mddpg/vddpg/ant/" + exp_index
mode = "ec2"
ec2_instance = "c4.2xlarge"
subnet = "us-west-1c"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3" # needs psutils
config.AWS_IMAGE_ID = "ami-85d181e5" # with docker already pulled

n_task_per_instance = 1
n_parallel = 2 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 10
plot = False

# variant params ---------------------------------------------------
class VG(VariantGenerator):
    @variant
    def zzseed(self):
        return [0, 100, 200, 300, 400]

    @variant
    def env_name(self):
        return ["ant_puddle"]
    @variant
    def maze(self):
        return [
            dict(wall_offset=0.25, u_length=3, u_turn_length=7),
        ]
    @variant
    def init_reward(self):
        return [0.1]
    @variant
    def goal_reward(self):
        return [10]
    @variant
    def speed_coeff(self):
        return [0]

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
    @variant
    def network_size(self):
        return [200]
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


variants = VG().variants()
batch_tasks = []
print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params -----------------------------------
    # >>>>>>
    # Plotter settings.
    q_plot_settings = None

    env_plot_settings = dict()
    # algo
    seed=v["zzseed"]
    env_name = v["env_name"]
    shared_ddpg_kwargs = dict(
        max_path_length=v["max_path_length"],
        scale_reward=v["scale_reward"],
        qf_learning_rate=1e-3,
        soft_target_tau=1.,
        actor_train_frequency=v["train_frequency"]["actor_train_frequency"],
        critic_train_frequency=v["train_frequency"]["critic_train_frequency"],
        update_target_frequency=v["train_frequency"]["update_target_frequency"],
        train_repeat=v["train_frequency"]["train_repeat"],
        env_plot_settings=env_plot_settings,
        q_plot_settings=q_plot_settings,
        debug_mode=False,
    )
    if "local" in mode and sys.platform == 'darwin':
        shared_ddpg_kwargs["plt_backend"] = "MacOSX"
    else:
        shared_ddpg_kwargs["plt_backend"] = "Agg"

    if mode == "local_test" or mode == "local_docker_test":
        ddpg_kwargs = dict(
            epoch_length = 200,
            min_pool_size = 100,
            eval_samples = 100,
            n_epochs = 2,
        )
    else:
        ddpg_kwargs = dict(
            epoch_length=10000,
            n_epochs=2000,
            eval_samples=v["max_path_length"] * 10,
                # deterministic env and policy: only need 1 traj sample
        )
    ddpg_kwargs.update(shared_ddpg_kwargs)

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
    G = AntPuddleGenerator()
    env_kwargs = {
        "reward_type": "goal",
        "init_reward": v["init_reward"],
        "goal_reward": v["goal_reward"],
        "speed_coeff": v["speed_coeff"],
        "mujoco_env_args": {
            "random_init_state": False,
        },
        "puddles": G.generate_u_shaped_maze(
            wall_offset=v["maze"]["wall_offset"],
            length=v["maze"]["u_length"],
            turn_length=v["maze"]["u_turn_length"],
            obj="puddles",
        ),
        "goal": G.generate_u_shaped_maze(
            wall_offset=v["maze"]["wall_offset"],
            length=v["maze"]["u_length"],
            turn_length=v["maze"]["u_turn_length"],
            obj="goal",
        ),
        "plot_settings": G.generate_u_shaped_maze(
            wall_offset=v["maze"]["wall_offset"],
            length=v["maze"]["u_length"],
            turn_length=v["maze"]["u_turn_length"],
            obj="plot_settings",
        ),
    }
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
        embedded_hidden_sizes=(v["network_size"], v["network_size"]),
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
        observation_hidden_sizes=(v["network_size"], v["network_size"]),
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
        )
        batch_tasks = []
        if "test" in mode:
            sys.exit(0)
if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
