from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv

from sandbox.haoran.mddpg.envs.mujoco.swimmer_undirected_env import \
    SwimmerUndirectedEnv
from sandbox.haoran.mddpg.envs.double_slit_env import DoubleSlitEnv

class EnvChooser(object):
    def __init__(self):
        pass

    def choose_env(self,env_name,**kwargs):
        if env_name == "swimmer":
            env = SwimmerEnv(**kwargs)
        elif env_name == "swimmer_undirected":
            env = SwimmerUndirectedEnv(**kwargs)
        elif env_name == "hopper":
            env = HopperEnv(**kwargs)
        elif env_name == "walker":
            env = Walker2DEnv(**kwargs)
        elif env_name == "halfcheetah":
            env = HalfCheetahEnv(**kwargs)
        elif env_name == "ant":
            env = AntEnv(**kwargs)
        elif env_name == "humanoid":
            env = SimpleHumanoidEnv(**kwargs)
        elif env_name == "cartpole":
            env = CartpoleEnv(**kwargs)
        elif env_name == "double_pendulum":
            env = DoublePendulumEnv(**kwargs)
        elif env_name == "inv_double_pendulum":
            env = InvertedDoublePendulumEnv(**kwargs)
        elif env_name == "double_slit":
            env = DoubleSlitEnv()
        else:
            print("Unknown environment %s"%(env_name))
            raise NotImplementedError
        return env
