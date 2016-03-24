import numpy as np
from nose2 import tools

from rllab.envs.box2d.car_parking_env import CarParkingEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.envs.identification_env import IdentificationEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.point_env import PointEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.noisy_env import NoisyObservationEnv, DelayedActionEnv
from rllab.envs.normalized_env import NormalizedEnv
from rllab.envs.proxy_env import ProxyEnv

simple_env_classes = [
    GridWorldEnv,
    CartpoleEnv,
    CarParkingEnv,
    CartpoleSwingupEnv,
    DoublePendulumEnv,
    MountainCarEnv,
    PointEnv,
    Walker2DEnv,
    SwimmerEnv,
    SimpleHumanoidEnv,
    InvertedDoublePendulumEnv,
    HopperEnv,
    HalfCheetahEnv,
]
envs = [cls() for cls in simple_env_classes]
envs.append(
    ProxyEnv(envs[0])
)
envs.append(
    IdentificationEnv(CartpoleEnv, {})
)
envs.append(
    NoisyObservationEnv(CartpoleEnv())
)
envs.append(
    DelayedActionEnv(CartpoleEnv())
)
envs.append(
    NormalizedEnv(CartpoleEnv())
)


@tools.params(*envs)
def test_env(env):
    print "Testing", env.__class__
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob)
    a = act_space.sample()
    assert act_space.contains(a)
    res = env.step(a)
    assert ob_space.contains(res.observation)
    assert np.isscalar(res.reward)
    env.render()

