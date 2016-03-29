from rllab.misc import instrument
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
from rllab.envs.normalized_env import normalize
from sandbox.rocky.dpg.dpg import DPG
from sandbox.rocky.dpg.continuous_mlp_policy import ContinuousMLPPolicy
from sandbox.rocky.dpg.continuous_mlp_q_function import ContinuousMLPQFunction
from sandbox.rocky.dpg.ou_strategy import OUStrategy

instrument.stub(globals())

vg = instrument.VariantGenerator()
vg.add("seed", [1, 11, 21, 31, 41])
vg.add("env", map(normalize, [
    CartpoleEnv(),
    CartpoleSwingupEnv(),
    MountainCarEnv(),
    DoublePendulumEnv(),
    SwimmerEnv(),
    HopperEnv(),
    Walker2DEnv(),
    HalfCheetahEnv(),
    AntEnv(),
    InvertedDoublePendulumEnv(),
    SimpleHumanoidEnv(),
    HumanoidEnv(),
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
    algo = DPG(
        qf_learning_rate=variant["qf_lr"],
        qf_weight_decay=variant["qf_weight_decay"],
        policy_learning_rate=variant["policy_lr"],
        max_path_length=500,
        epoch_length=10000,
        n_epochs=2500,
        scale_reward=variant["scale_reward"],
        soft_target_tau=variant["soft_target_tau"],
        eval_samples=10000,
        # n_epochs=50,
    )
    policy = ContinuousMLPPolicy(
        env_spec=variant["env"].spec,
        hidden_sizes=(400, 300),
        bn=variant["bn"],
    )
    qf = ContinuousMLPQFunction(
        env_spec=variant["env"].spec,
        hidden_sizes=(400, 300),
        bn=variant["bn"],
    )
    instrument.run_experiment_lite(
        algo.train(env=variant["env"], policy=policy, qf=qf, es=variant["es"]),
        exp_prefix="dpg_all_search_fixed",
        n_parallel=4,
        snapshot_mode="last",
        seed=variant["seed"],
        mode="ec2",
        # dry=True,
    )
    # break
