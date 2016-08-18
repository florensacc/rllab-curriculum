# RAM version of DQN (no Montezuma)

from __future__ import print_function
from __future__ import absolute_import

import sys,os
sys.path.append('.')
import numpy as np
import theano
import json

from rllab import config
os.path.join(config.PROJECT_PATH)
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.haoran.deep_q_rl.deep_q_rl.ale_agent import NeuralAgent
from sandbox.haoran.deep_q_rl.deep_q_rl.ale_experiment import ALEExperiment
from sandbox.haoran.deep_q_rl.deep_q_rl.q_network import DeepQLearner
from sandbox.haoran.hashing.hash.sim_hash import SimHash
from sandbox.haoran.hashing.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.haoran.hashing.preprocessor.image_vectorize_preprocessor import ImageVectorizePreprocessor

stub(globals())

# check for uncommitted changes ------------------------------
# import git
# repo = git.Repo('.')
# if repo.is_dirty():
#     answer = ''
#     while answer not in ['y','Y','n','N']:
#         answer = raw_input("The repository has uncommitted changes. Do you want to continue? (y/n)")
#     if answer in ['n','N']:
#         sys.exit(1)

# define running mode specific params -----------------------------------
exp_prefix = "hashing/" + os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "ec2_cpu"
snapshot_mode = "all"
plot = False
use_gpu = False # should change conv_type and ~/.theanorc

# config.DOCKER_IMAGE = 'tsukuyomi2044/rllab'
if "ec2_cpu" in mode:
    config.AWS_INSTANCE_TYPE = "m4.large"
    config.AWS_SPOT_PRICE = '0.1'
    config.DOCKER_IMAGE = "dementrock/rllab-shared"
    plot = False
elif "ec2_gpu" in mode:
    config.AWS_INSTANCE_TYPE = "g2.2xlarge"
    config.AWS_SPOT_PRICE = '0.5'
    config.DOCKER_IMAGE = "tsukuyomi2044/rllab_gpu"
    plot = False

# different training params ------------------------------------------
from rllab.misc.instrument import VariantGenerator, variant
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1,101,201,301,401]

    @variant
    def bonus_coeff(self):
        return [0.01, 0.1, 1, 0]

    @variant
    def dim_key(self):
        return [128, 256, 512]

    @variant
    def game(self):
        return ["breakout", "freeway", "frostbite"]

    @variant
    def bucket_sizes(self):
        return [
            [15485867, 15485917, 15485927, 15485933, 15485941, 15485959],
            # should be more than enough; see https://primes.utm.edu/lists/small/millions/
            [999931, 999953, 999959, 999961, 999979, 999983],
        ]

variants = VG().variants()
exp_names = []
for v in variants:
    # define the exp_name (log folder name) -------------------
    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    exp_name = "alex_{time}_{game}".format(
        time=timestamp,
        game=v["game"],
    )
    exp_names.append(exp_name)
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    # parameters -----------------------------------------------
    # Experiment Parameters
    # ----------------------
    steps_per_epoch = 50000
    epochs = 500
    steps_per_test = 10000

    observation_type = "ram"
    record_ram = True
    record_image = False

    # ----------------------
    # ale parameters
    # ----------------------
    base_rom_path = "sandbox/haoran/deep_q_rl/roms/"
    rom = '%s.bin'%(v["game"])
    frame_skip = 4
    repeat_action_probability = 0

    # ----------------------
    # agent/network parameters:
    # ----------------------

    update_rule = 'deepmind_rmsprop'
    batch_accumulator = 'sum'
    learning_rate = .00025
    discount = .99
    rms_decay = .95 # (rho)
    rms_epsilon = .01
    momentum = 0
    clip_delta = 0
    epsilon_start = 1.0
    epsilon_min = .1
    epsilon_decay = 1000000
    update_frequency = 4 # update q-values for every this many time steps
    replay_memory_size = 1000000
    batch_size = 32
    resize_method = 'crop'
    resized_width = 84
    resized_height = 84
    network_type = "mlp" # the network is smaller
    hidden_sizes = [32,32]
    conv_type = ""
    freeze_interval = 10000
    replay_start_size = 30000
    death_ends_episode = False # run_nature uses True
    max_start_nullops = 30
    cudnn_deterministic = False # setting True can result in error
    use_double = True
    clip_reward = True
    agent_unpicklable_list = ["data_set","test_data_set","bonus_evaluator"]
    seed = v["seed"]
    recording = False
    max_episode_length = np.inf

    if observation_type == "image":
        input_width = resized_width
        input_height = resized_height
        phi_length = 4
    else:
        input_width = 128
        input_height = 1
        phi_length = 1


    if cudnn_deterministic:
        theano.config.dnn.conv.algo_bwd = 'deterministic'
    freeze_interval = freeze_interval // update_frequency # see launcher.py for a reason

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

    # load game specific information -----------------------------------
    game_info_path = os.path.join(base_rom_path, v["game"] + "_info.json")
    with open(game_info_path) as game_info_file:
        game_info = json.load(game_info_file)
    min_action_set_length = game_info["min_action_set_length"]


    state_preprocessor = ImageVectorizePreprocessor(
        n_chanllel=phi_length,
        width=input_width,
        height=input_height,
    )
    hash_list = [
        SimHash(
            item_dim=state_preprocessor.get_output_dim(), # get around stub
            dim_key=v["dim_key"],
            bucket_sizes=v["bucket_sizes"],
        )
    ]
    bonus_evaluator = ALEHashingBonusEvaluator(
        state_dim=state_preprocessor.get_output_dim(),
        state_preprocessor=state_preprocessor,
        num_actions=min_action_set_length,
        hash_list=hash_list,
        count_mode="s",
        bonus_mode="s_next",
        bonus_coeff=v["bonus_coeff"],
        state_bonus_mode="1/n_s",
        state_action_bonus_mode="log(n_s)/n_sa",
    )
    q_network = DeepQLearner(
        input_width=input_width,
        input_height=input_height,
        num_actions=min_action_set_length,
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
        hidden_sizes=hidden_sizes,
        )

    agent = NeuralAgent(
        q_network=q_network,
        bonus_evaluator=bonus_evaluator,
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
        seed=v["seed"],
        n_parallel=1, # we actually don't use parallel_sampler here
        snapshot_mode=snapshot_mode,
        mode=actual_mode,
        variant=v,
        terminate_machine=terminate_machine,
        use_gpu=use_gpu,
    )
    if "test" in mode:
        sys.exit(0)

# logging -------------------------------------------------------------
# record the experiment names to a file
# also record the branch name and commit number
# logs = []
# logs += ["branch: %s" %(repo.active_branch.name)]
# logs += ["commit SHA: %s"%(repo.head.object.hexsha)]
# logs += exp_names
#
# cur_script_name = __file__
# log_file_name = cur_script_name.split('.py')[0] + '.log'
# with open(log_file_name,'w') as f:
#     for message in logs:
#         f.write(message + "\n")

# make the current script read-only to avoid accidental changes after ec2 runs
if "local" not in mode:
    os.system("chmod 444 %s"%(__file__))
