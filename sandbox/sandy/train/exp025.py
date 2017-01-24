# imports -----------------------------------------------------
""" agents """
from sandbox.sandy.async_rl.agents.a3c_agent import A3CAgent
#from sandbox.sandy.async_rl.agents.dqn_agent import DQNAgent

""" algorithm """
from sandbox.sandy.async_rl.algos.a3c_ale import A3CALE
# Don't use asynchronous DQN -- apparently it doesn't work
#from sandbox.sandy.async_rl.algos.dqn_ale import DQNALE

""" environment """
from sandbox.sandy.envs.atari_env_a3c import AtariEnvA3C
from sandbox.sandy.async_rl.envs.atari_env import AtariEnv

""" others """
from sandbox.sandy.misc.util import get_time_stamp
from sandbox.sandy.misc.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import numpy as np
import sys,os
import logging

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup -----------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "ec2"  # "local_docker"
ec2_instance = "c4.8xlarge"
price_multiplier = 3
subnet = "us-west-1a"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3:theano" # needs psutils
logging_level = logging.INFO

n_parallel = 2 # only for local exp
snapshot_mode = "gap"
snapshot_gap = 20
plot = False
use_gpu = False
sync_s3_pkl = True
sync_s3_log = True
config.USE_TF = False

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0, 100, 200, 300, 400]

    @variant
    def frame_skip(self):
        return [4]

    @variant
    def img_size(self):
        return [84]

    @variant
    def game(self):
        return ["space_invaders"]

variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    # non-variant params ------------------------------
    # algo
    use_async = True
    seed = v["seed"]
    persistent = True
    img_size = v["img_size"]
    seeds = None

    # env
    game = v["game"]
    exp_prefix = "async-rl/" + exp_index + "-" + game.split('_')[0]
    env_seed = 1 # deterministic env
    frame_skip = v['frame_skip']
    img_width = img_size
    img_height = img_size
    n_last_screens = 4
    n_iter = 200

    # problem setting
    agent_type = "a3c"
    entropy_bonus = 0.01
    shared_weights = True
    t_max = 5
    lr = 7e-4
    sync_t_gap_limit = 1000
    eval_frequency = 10**6
    eval_n_runs = 10
    target_update_frequency = 40000
    record_ram = False

    if "test" in mode:
        eval_frequency = 2000
        eval_n_runs = 1
        target_update_frequency = 400

    # other exp setup --------------------------------------
    exp_name = "{exp_index}_{time}_{game}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        game=game,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    if use_gpu:
        config.USE_GPU = True
        config.DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"

    if "local_docker" in mode:
        actual_mode = "local_docker"
    elif "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
        # configure instance
        info = instance_info[ec2_instance]
        config.AWS_INSTANCE_TYPE = ec2_instance
        config.AWS_SPOT_PRICE = str(info["price"] * price_multiplier)
        if config.AWS_REGION_NAME == "us-west-1":
             config.AWS_IMAGE_ID = "ami-271b4847"  # Use Haoran's AWS image with his docker iamge

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
    else:
        raise NotImplementedError

    # construct objects ----------------------------------
    assert game in ["pong", "space_invaders", "beamrider", "breakout", "qbert", \
                    "chopper_command", "seaquest", "skiing"], "Invalid game choice"
    if game == 'pong':
        gym_game = 'Pong-v3'
    elif game == "space_invaders":
        gym_game = 'SpaceInvaders-v3'
    elif game == "beamrider":
        gym_game = 'BeamRider-v3'
    elif game == "breakout":
        gym_game = 'Breakout-v3'
    elif game == "qbert":
        gym_game = 'Qbert-v3'
    elif game == "chopper_command":
        gym_game = 'ChopperCommand-v3'
    elif game == "seaquest":
        gym_game = "Seaquest-v3"
    elif game == "skiing":
        gym_game = 'Skiing-v3'
    else:
        assert False, "Invalid game choice"

    #env = AtariEnvA3C(
    #        gym_game,
    #        force_reset=True,
    #        seed=env_seed,
    #        img_width=img_width,
    #        img_height=img_height,
    #        n_frames=n_last_screens,
    #        frame_skip=frame_skip,
    #        persistent=persistent
    #)

    env = AtariEnv(
        game=game,
        rom_filename="",
        plot=plot,
        record_ram=record_ram,
    )

    if agent_type == "a3c":
        agent = A3CAgent(
            t_max=t_max,    
            n_actions=env.number_of_actions,
	    beta=entropy_bonus,
            sync_t_gap_limit=sync_t_gap_limit,
            optimizer_args=dict(
                lr=lr,
                eps=1e-1,
                alpha=0.99
            ),
            shared_weights=shared_weights,
            img_size=img_size
        )
        algo = A3CALE(
            n_processes=n_parallel,
            env=env,
            agent=agent,
            logging_level=logging_level,
            total_steps=n_iter*eval_frequency,
            eval_frequency=eval_frequency,
            eval_n_runs=eval_n_runs,
            seeds=seeds,
        )
    #elif agent_type == "dqn":
    #    agent = DQNAgent(
    #        t_max=t_max,
    #        n_actions=env.number_of_actions,
    #        target_update_frequency=target_update_frequency,
    #        eps_test=eps_test,
    #        eps_anneal_time=eps_anneal_time,
    #        eps_end=eps_end,
    #        sync_t_gap_limit=sync_t_gap_limit,
    #        optimizer_args=dict(
    #            lr=lr,
    #            eps=1e-1,
    #            alpha=0.99
    #        )
    #    )
    #    algo = DQNALE(
    #        n_processes=n_parallel,
    #        env=env,
    #        agent=agent,
    #        logging_level=logging_level,
    #        eval_frequency=eval_frequency,
    #        eval_n_runs=eval_n_runs,
    #        seeds=seeds,
    #    )

    if use_async:
        run_experiment_lite(
            algo.train(),
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            seed=seed,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            mode=actual_mode,
            variant=v,
            use_gpu=use_gpu,
            plot=plot,
            sync_s3_pkl=sync_s3_pkl,
            sync_s3_log=sync_s3_log,
            sync_log_on_termination=True,
            sync_all_data_node_to_s3=True,
        )
    else:
        raise NotImplementedError

    if "test" in mode:
        sys.exit(0)

if ("local" not in mode) and ("test" not in mode):
    os.system("chmod 444 %s"%(__file__))
