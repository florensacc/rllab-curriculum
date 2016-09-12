"""
Try disabling entropy bonus in freeway
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
mode = "ec2"
ec2_instance = "c4.8xlarge"
subnet = "us-west-1a"
repetitions = 1 # each repetition uses a different set of random seeds
logging_level = logging.INFO


# Generic setting
rom_dir = "ale_python_interface/roms"
exp_prefix = "async-rl/" + os.path.basename(__file__).split('.')[0]
snapshot_mode = "last"
plot = False
seeds = None
n_processes = 18

# Problem setting
eval_frequency = 10**5
eval_n_runs = 10
games = ["freeway"]
agent_types = ["a3c"]
bonus_coeffs = [0.05]
entropy_bonus = 0
dim_key = 64
target_update_frequency = 40000
locked_stats = True

for bonus_count_target in ["ram","image"]:
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
    if "test" in mode:
        eval_frequency = 200
        eval_n_runs = 1
        target_update_frequency = 400

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

    for idx,game,agent_type,bonus_coeff in \
        itertools.product(range(repetitions),games,agent_types,bonus_coeffs):
        # The meat ---------------------------------------------
        env = AtariEnv(
            rom_filename=os.path.join(rom_dir,game+".bin"),
            plot=plot,
            record_ram=record_ram,
        )
        _hash = SimHash(
            item_dim=count_target_dim,
            dim_key=dim_key,
            bucket_sizes=None,
        )
        bonus_evaluator = ALEHashingBonusEvaluator(
            state_dim=count_target_dim,
            state_preprocessor=preprocessor,
            hash=_hash,
            bonus_coeff=bonus_coeff,
            state_bonus_mode="1/sqrt(n_s)",
            log_prefix="",
            locked_stats=locked_stats,
        )

        if agent_type == "a3c":
            agent = A3CAgent(
                n_actions=env.number_of_actions,
                bonus_evaluator=bonus_evaluator,
                bonus_count_target=bonus_count_target,
		beta=entropy_bonus,
            )
            algo = A3CALE(
                n_processes=n_processes,
                env=env,
                agent=agent,
                logging_level=logging_level,
                eval_frequency=eval_frequency,
                eval_n_runs=eval_n_runs,
                seeds=seeds,
            )
        elif agent_type == "dqn":
            agent = DQNAgent(
                n_actions=env.number_of_actions,
                bonus_evaluator=bonus_evaluator,
                bonus_count_target=bonus_count_target,
                target_update_frequency=target_update_frequency,
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



        # Exp config --------------------------------------------------
        # Exp name
        import datetime
        import dateutil.tz
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        exp_name = "alex_{time}_{agent_type}_{game}".format(
            time=timestamp,
            agent_type=agent_type,
            game=game,
        )
        if ("ec2" in mode) and (len(exp_name) > 64):
            print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
            sys.exit(1)


        terminate_machine = not ("test" in mode)

        run_experiment_lite(
            algo.train(),
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            snapshot_mode=snapshot_mode,
            mode=actual_mode,
            sync_s3_pkl=True,
            terminate_machine=terminate_machine,
        )

        if "test" in mode:
            sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
