"""
Test whatever you want here
"""

import logging
import os,sys
import numpy as np
import itertools

from rllab.envs.sliding_mem_env import SlidingMemEnv
from sandbox.pchen.dqn.envs.atari import AtariEnvCX

sys.path.append('.')

from sandbox.pchen.async_rl.async_rl.agents.a3c_agent import A3CAgent
from sandbox.pchen.async_rl.async_rl.agents.dqn_agent import DQNAgent
from sandbox.pchen.async_rl.async_rl.algos.a3c_ale import A3CALE
from sandbox.pchen.async_rl.async_rl.algos.dqn_ale import DQNALE
from sandbox.pchen.async_rl.async_rl.utils.get_time_stamp import get_time_stamp
from sandbox.pchen.async_rl.async_rl.utils.ec2_instance import instance_info, subnet_info
from sandbox.pchen.async_rl.async_rl.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.pchen.async_rl.async_rl.preprocessor.image_vectorize_preprocessor import ImageVectorizePreprocessor
from sandbox.pchen.async_rl.async_rl.hash.sim_hash import SimHash

from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, stub
from rllab import config

# stub(globals())

mode = "local"
# mode = "local_docker"

ec2_instance = "c4.8xlarge"
logging_level = logging.INFO


# Generic setting
rom_dir = "ale_python_interface/roms"
exp_prefix = "async-rl/" + os.path.basename(__file__).split('.')[0]
snapshot_mode = "last"
plot = False
seeds = None
n_processes = 2

# Problem setting
game = "frostbite"
agent_type = "dqn"
entropy_bonus = 0.01
bonus_coeff = 0.05
bonus_count_target = "ram"
dim_key = 64
target_update_frequency = 40000
locked_stats = True
eps_test = 0.01

eval_frequency = 2000000000000000000000000000000
eval_n_runs = 1

# Secondary problem setting
if bonus_count_target == "ram":
    record_ram = True
    count_target_dim = 128
    preprocessor = None
else:
    record_ram = False
    n_channel = 4
    img_width = 84
    img_height = 84
    count_target_dim = n_channel * img_width * img_height
    preprocessor = ImageVectorizePreprocessor(
        n_channel=n_channel,
        width=img_width,
        height=img_height,
    )

# Other configuration stuffs
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
    n_processes = info["vCPU"]

    # choose subnet
    config.AWS_NETWORK_INTERFACES = [
        dict(
            SubnetId=subnet_info[subnet]["SubnetID"],
            Groups=subnet_info[subnet]["Groups"],
            DeviceIndex=0,
            AssociatePublicIpAddress=True,
        )
    ]
else:
    raise NotImplementedError

# The meat ---------------------------------------------
# env = AtariEnv(
#     rom_filename=os.path.join(rom_dir,game+".bin"),
#     plot=plot,
#     record_ram=record_ram,
# )
env = AtariEnvCX(obs_type="image")
env = SlidingMemEnv(env)

agent = DQNAgent(
    # n_actions=env.number_of_actions,
    env=env,
    target_update_frequency=target_update_frequency,
    eps_test=eps_test,
)
algo = DQNALE(
    n_processes=n_processes,
    env=env,
    agent=agent,
    logging_level=logging_level,
    eval_frequency=eval_frequency,
    eval_n_runs=eval_n_runs,
    seeds=seeds,
)
algo.train()

#
#
# # Exp config --------------------------------------------------
# # Exp name
# exp_name = "pchen_{time}_{agent_type}_{game}".format(
#     time=get_time_stamp(),
#     agent_type=agent_type,
#     game=game,
# )
#
# terminate_machine = not ("test" in mode)
#
# run_experiment_lite(
#     algo.train(),
#     exp_prefix=exp_prefix,
#     exp_name=exp_name,
#     snapshot_mode=snapshot_mode,
#     mode="local",
# )

