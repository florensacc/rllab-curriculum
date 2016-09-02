from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv


from rllab.envs.box2d.car_parking_env import CarParkingEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv

env_names = [
    "swimmer",
    "hopper",
    "halfcheetah",
    "walker",
    "car_parking",
    "cartpole_swingup",
    "mountain_car",
    "double_pendulum",
    "cartpole",
    "ant",
    "human",
]    


for env_name in env_names:
    if env_name == "swimmer":
        env = SwimmerEnv()
    elif env_name == "hopper":
        env = HopperEnv()
    elif env_name == "halfcheetah":
        env = HalfCheetahEnv()
    elif env_name == "walker":
        env = Walker2DEnv()
    elif env_name == "car_parking":
        env = CarParkingEnv()
    elif env_name == "cartpole_swingup":
        env = CartpoleSwingupEnv()
    elif env_name == "mountain_car": 
        env = MountainCarEnv()
    elif env_name in ["double_pendulum","dpend"]:
        env = DoublePendulumEnv()
    elif env_name == "cartpole":
        env = CartpoleEnv()
    elif env_name == "ant":
        env = AntEnv()
    elif env_name == "human":
        env = SimpleHumanoidEnv()
    else: 
        raise NotImplementedError
    
    print "%-20s %d"%(env_name,len(env.action_bounds[0]))

