class EnvChooser(object):
    def __init__(self):
        pass

    def choose_env(self,env_name,**kwargs):
        if env_name == "swimmer":
            from rllab.envs.mujoco.swimmer_env import SwimmerEnv
            env = SwimmerEnv(**kwargs)

        elif env_name == "swimmer_undirected":
            from sandbox.haoran.mddpg.envs.mujoco.swimmer_undirected_env \
                import SwimmerUndirectedEnv
            env = SwimmerUndirectedEnv(**kwargs)

        elif env_name == "hopper":
            from rllab.envs.mujoco.hopper_env import HopperEnv
            env = HopperEnv(**kwargs)

        elif env_name == "walker":
            from rllab.envs.mujoco.walker2d_env import Walker2DEnv
            env = Walker2DEnv(**kwargs)

        elif env_name == "halfcheetah":
            from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
            env = HalfCheetahEnv(**kwargs)

        elif env_name == "ant":
            from rllab.envs.mujoco.ant_env import AntEnv
            env = AntEnv(**kwargs)

        elif env_name == "humanoid":
            from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
            env = SimpleHumanoidEnv(**kwargs)

        elif env_name == "cartpole":
            from rllab.envs.box2d.cartpole_env import CartpoleEnv
            env = CartpoleEnv(**kwargs)

        elif env_name == "double_pendulum":
            from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
            env = DoublePendulumEnv(**kwargs)

        elif env_name == "inv_double_pendulum":
            from rllab.envs.mujoco.inverted_double_pendulum_env import \
                InvertedDoublePendulumEnv
            env = InvertedDoublePendulumEnv(**kwargs)

        elif env_name == "double_slit":
            from sandbox.haoran.mddpg.envs.double_slit_env import DoubleSlitEnv
            env = DoubleSlitEnv()

        elif env_name == "double_slit_v2":
            from sandbox.haoran.mddpg.envs.double_slit_env_v2 import \
                DoubleSlitEnvV2
            env = DoubleSlitEnvV2()
        elif env_name == "multi_goal":
            from sandbox.haoran.mddpg.envs.multi_goal_env import \
                MultiGoalEnv
            env = MultiGoalEnv()
        else:
            print("Unknown environment %s"%(env_name))
            raise NotImplementedError
        return env
