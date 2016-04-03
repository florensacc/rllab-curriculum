from rllab.algos.ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv

from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from rllab.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv

from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc import instrument
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

instrument.stub(globals())

vg = instrument.VariantGenerator()
vg.add("seed", [1, 11, 21, 31, 41])
vg.add("env", map(normalize, [
    AntGatherEnv(),
    SwimmerGatherEnv(),
    AntMazeEnv(),
    SwimmerMazeEnv(),

    # CartpoleEnv(),
    # CartpoleSwingupEnv(),
    # MountainCarEnv(),
    # DoublePendulumEnv(),
    # SwimmerEnv(),
    # HopperEnv(),
    # Walker2DEnv(),
    # HalfCheetahEnv(),
    # AntEnv(),
    # InvertedDoublePendulumEnv(),
    # SimpleHumanoidEnv(),
    # HumanoidEnv(),
]))
vg.add("es", lambda env: [OUStrategy(env_spec=env.spec, theta=0.15, sigma=0.3)])
vg.add("qf_weight_decay", [0.])  # , 0.01])
vg.add("qf_lr", [1e-3])  # , 1e-4, 1e-5])
vg.add("scale_reward", [0.1])  # 1., 0.1, 0.01, 0.001])
vg.add("policy_lr", [1e-4])  # 5, 1e-4, 1e-3])
vg.add("soft_target_tau", [1e-3])  # , 1e-4])
vg.add("bn", [False])

print "#Experiments:", len(vg.variants())
for variant in vg.variants():
    policy = DeterministicMLPPolicy(
        env_spec=variant["env"].spec,
        hidden_sizes=(400, 300),
        bn=variant["bn"],
    )
    qf = ContinuousMLPQFunction(
        env_spec=variant["env"].spec,
        hidden_sizes=(400, 300),
        bn=variant["bn"],
    )
    algo = DDPG(
        env=variant["env"],
        policy=policy,
        qf=qf,
        es=variant["es"],
        qf_learning_rate=variant["qf_lr"],
        qf_weight_decay=variant["qf_weight_decay"],
        policy_learning_rate=variant["policy_lr"],
        max_path_length=500,
        min_pool_size=10000,
        epoch_length=10000,
        n_epochs=2500,
        scale_reward=variant["scale_reward"],
        soft_target_tau=variant["soft_target_tau"],
        eval_samples=10000,
        # n_epochs=50,
    )
    instrument.run_experiment_lite(
        algo.train(),
        exp_prefix="dpg_new_search",
        n_parallel=4,
        snapshot_mode="last",
        seed=variant["seed"],
        mode="ec2",
        # use_gpu=True,
        # env=dict(OMP_NUM_THREADS=1),
        # terminate_machine=False,
        # dry=True,
    )
    # break
    # break
