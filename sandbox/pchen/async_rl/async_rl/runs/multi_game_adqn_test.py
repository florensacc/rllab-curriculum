"""
Test whatever you want here
"""

import logging
import os,sys
import numpy as np
import itertools
sys.path.append('.')

from sandbox.pchen.async_rl.async_rl.agents.a3c_agent import A3CAgent
from sandbox.pchen.async_rl.async_rl.agents.dqn_agent import DQNAgent
from sandbox.pchen.async_rl.async_rl.envs.atari_env import AtariEnv
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

stub(globals())

# Running platform

# Generic setting
rom_dir = "ale_python_interface/roms"
exp_prefix = "async-rl/" + os.path.basename(__file__).split('.')[0]
snapshot_mode = "last"
plot = False
n_processes = 18

# Problem setting
game = "frostbite"
target_update_frequency = 40000
locked_stats = True
eps_test = 0.01

eval_frequency = 420000 # roughly every 5 minutes
eval_n_runs = 20

# The meat ---------------------------------------------
env = AtariEnv(
    rom_filename=os.path.join(rom_dir,game+".bin"),
    plot=plot,
)

agent = DQNAgent(
    n_actions=env.number_of_actions,
    target_update_frequency=target_update_frequency,
    eps_test=eps_test,
)
algo = DQNALE(
    n_processes=n_processes,
    env=env,
    agent=agent,
    eval_frequency=eval_frequency,
    eval_n_runs=eval_n_runs,
)


# Exp config --------------------------------------------------
# Exp name
exp_name = "pchen_adqn_{game}".format(
    # time=get_time_stamp(),
    # agent_type=agent_type,
    game=game,
)

run_experiment_lite(
    algo.train(),
    exp_prefix=exp_name,
    exp_name=exp_name,
    snapshot_mode=snapshot_mode,
    mode="lab_kube",
)
