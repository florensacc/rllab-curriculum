"""
Test run with the following changes
* MC return as part of Q target
* nips network
"""

# imports -----------------------------------------------------
""" agents """
from sandbox.sandy.deep_q_rl.ale_agent import NeuralAgent
from sandbox.sandy.deep_q_rl.q_network import DeepQLearner

""" algorithm """
from sandbox.sandy.deep_q_rl.ale_experiment import ALEExperiment

""" environment """
from sandbox.sandy.envs.atari_env_dqn import AtariEnvDQN

""" others """
from sandbox.sandy.misc.util import get_time_stamp
from sandbox.sandy.misc.ec2_info import instance_info, subnet_info
from rllab import config
from rllab.misc.instrument import stub, run_experiment_lite
import numpy as np
import sys,os
import theano
import json

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# exp setup -----------------------------------------------------
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "ec2_gpu"
#mode = "local_docker_gpu_test"
ec2_instance = "p2.xlarge"
price_multiplier = 3
subnet = "us-west-2b"
config.DOCKER_IMAGE = "tsukuyomi2044/rllab3:theano" # needs psutils

snapshot_mode = "gap"
snapshot_gap = 20
if "test" in mode:
    snapshot_gap = 2
plot = False
use_gpu = "gpu" in mode
sync_s3_pkl = True
sync_s3_log = True
config.USE_TF = False

if "ec2" in mode:
    info = instance_info[ec2_instance]
    config.AWS_INSTANCE_TYPE = ec2_instance
    config.AWS_SPOT_PRICE = str(info["price"] * price_multiplier)
    plot = False

    # choose subnet
    config.AWS_NETWORK_INTERFACES = [
        dict(
            SubnetId=subnet_info[subnet]["SubnetID"],
            Groups=subnet_info[subnet]["Groups"],
            DeviceIndex=0,
            AssociatePublicIpAddress=True,
        )
    ]

if "ec2_cpu" in mode:
    n_parallel = int(info["vCPU"] /2)
elif "ec2_gpu" in mode or "docker" in mode:
    # config.DOCKER_IMAGE = "tsukuyomi2044/rllab_gpu"
    #config.DOCKER_IMAGE = "rein/rllab-exp-gpu"
    config.DOCKER_IMAGE = "shhuang/rllab-gpu"
    print("docker image:", config.DOCKER_IMAGE)
    if subnet.startswith("us-west-1"):
        config.AWS_REGION_NAME="us-west-1"
        config.AWS_IMAGE_ID = "ami-931a51f3"
    elif subnet.startswith("us-west-2"):
        config.AWS_REGION_NAME="us-west-2"
        config.AWS_IMAGE_ID = "ami-9af95dfa"
    else:
        raise NotImplementedError
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [0, 100, 200, 300, 400]

    @variant
    def death_ends_episode(self):
        return [False]

    @variant
    def game(self):
        return ["space_invaders"]

    @variant
    def eta(self):
        return [0]  # Used to be [0, 0.1] in Haoran's version

variants = VG().variants()

print("#Experiments: %d" % len(variants))
for v in variants:
    # env
    game = v['game']
    exp_prefix = "deep-q-rl/" + exp_index + "-" + game.split('_')[0]
    env_seed = 1 # deterministic env
    persistent = True
    scale_neg1_1 = False

    # other exp setup --------------------------------------
    exp_name = "{exp_index}_{time}_{game}".format(
        exp_index=exp_index,
        time=get_time_stamp(),
        game=game,
    )
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    # parameters -----------------------------------------------
    # ! -- consistent with Nature DQN
    # $ -- doesnot affect training
    # Experiment Parameters
    # ----------------------
    steps_per_epoch = 100000  # 10% of DQN's
    epochs = 500  # 10 times of DQN's
    steps_per_test = 10000  # $
    observation_type = "image"  # $
    record_ram = True # $
    record_image = True  # $

    # ----------------------
    # ale parameters
    # ----------------------
    base_rom_path = "/home/shhuang/anaconda2/envs/rllab3/lib/python3.5/site-packages/atari_py/atari_roms/"  # $ 
    rom = '%s.bin' % (game)  # $
    frame_skip = 4  # !
    repeat_action_probability = 0  # $
    conv_type = "cudnn"  # $

    # ----------------------
    # agent/network parameters:
    # ----------------------
    update_rule = 'deepmind_rmsprop'  # !
    batch_accumulator = 'sum'  # ?
    learning_rate = 0.00025  # !
    discount = .99  # !
    rms_decay = .95  # !
    rms_epsilon = .01  # !
    momentum = 0  # ?
    clip_delta = 1.0  # $ (see run_nature)
    epsilon_start = 1.0  # !
    epsilon_min = .1  # !
    epsilon_decay = 1000 * 1000  # !
    update_frequency = 4  # !
    replay_memory_size = 1000 * 1000  # !
    batch_size = 32  # !
    resize_method = 'scale'  # !
    resized_width = 84 # !
    resized_height = 84 # !
    network_type = "nips"  # !
    freeze_interval = 10000  # !
    replay_start_size = 50000  # !
    death_ends_episode = v["death_ends_episode"]  # Nature DQN uses True
    max_start_nullops = 0  # used to be 30 (!), but changed to 0 to be consistent with A3C and TRPO
    cudnn_deterministic = False  # $
    use_double = True  # !
    clip_reward = True  # !
    agent_unpicklable_list = ["data_set", "test_data_set", "bonus_evaluator"]  # $
    seed = v["seed"]  # $
    recording = False  # $
    max_episode_length = np.inf  # $

    bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
    network_args = dict()
    eta = v["eta"] # as in Bellemare's paper

    if observation_type == "image":
        input_width = resized_width
        input_height = resized_height
        phi_length = 4
    else:
        raise NotImplementedError

    if mode == "local_test" \
        or mode == "local_gpu_test" \
        or mode == "local_docker_test" \
        or mode == "local_docker_gpu_test":
        replay_start_size = 100
        steps_per_epoch = 1000
        steps_per_test = 4500
        epochs = 10

    if cudnn_deterministic:
        theano.config.dnn.conv.algo_bwd = 'deterministic'
    freeze_interval = freeze_interval // update_frequency  # see launcher.py for a reason

    # setup ALE (not stubbed) --------------------------------------------
    full_rom_path = os.path.join(base_rom_path, rom)
    if 'docker' in mode or 'ec2' in mode:
        full_rom_path = os.path.join(config.DOCKER_CODE_DIR, full_rom_path)
    ale_args = dict(
        seed=seed,
        plot=plot,
        repeat_action_probability=repeat_action_probability,
        rom_path=full_rom_path,
    )

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

    env = AtariEnvDQN(
        gym_game,
        force_reset=True,
        seed=env_seed,
        img_width=input_width,
        img_height=input_height,
        n_frames=phi_length,
        frame_skip=frame_skip,
        persistent=persistent,
        scale_neg1_1=scale_neg1_1
    )

    q_network = DeepQLearner(
        input_width=input_width,
        input_height=input_height,
        num_actions=env.number_of_actions,
        num_frames=phi_length,
        discount=discount,
        learning_rate=learning_rate,
        rho=rms_decay,
        rms_epsilon=rms_epsilon,
        momentum=momentum,
        clip_delta=clip_delta,
        freeze_interval=freeze_interval,
        use_double=use_double,
        batch_size=batch_size,
        network_type=network_type,
        conv_type=conv_type,
        update_rule=update_rule,
        batch_accumulator=batch_accumulator,
        input_scale=255.0,
        network_args=network_args,
        eta=eta,
    )

    agent = NeuralAgent(
        q_network=q_network,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        replay_memory_size=replay_memory_size,
        experiment_directory=None,
        replay_start_size=replay_start_size,
        update_frequency=update_frequency,
        clip_reward=clip_reward,
        recording=recording,
        unpicklable_list=agent_unpicklable_list,
    )

    experiment = ALEExperiment(
        ale_args=ale_args,
        agent=agent,
        env=env,
        resized_width=resized_width,
        resized_height=resized_height,
        resize_method=resize_method,
        num_epochs=epochs,
        epoch_length=steps_per_epoch,
        test_length=steps_per_test,
        frame_skip=frame_skip,
        death_ends_episode=death_ends_episode,
        max_start_nullops=max_start_nullops,
        max_episode_length=max_episode_length,
        observation_type=observation_type,
        record_image=record_image,
        record_ram=record_ram,
        game=game
    )

    # run --------------------------------------------------
    terminate_machine = "test" not in mode
    if "local_docker" in mode:
        actual_mode = "local_docker"
    elif "local" in mode:
        actual_mode = "local"
    elif "ec2" in mode:
        actual_mode = "ec2"
    else:
        raise NotImplementedError

    run_experiment_lite(
        stub_method_call=experiment.run(),
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        seed=seed,
        n_parallel=1,  # we actually don't use parallel_sampler here
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
        terminate_machine=terminate_machine,
    )
    if "test" in mode:
        sys.exit(0)

# make the current script read-only to avoid accidental changes after ec2 runs
if "local" not in mode:
    os.system("chmod 444 %s" % (__file__))
