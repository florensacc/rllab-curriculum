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
import git
repo = git.Repo('.')
if repo.is_dirty():
    answer = ''
    while answer not in ['y','Y','n','N']:
        answer = raw_input("The repository has uncommitted changes. Do you want to continue? (y/n)")
    if answer in ['n','N']:
        sys.exit(1)

# define running mode specific params -----------------------------------
exp_prefix = os.path.basename(__file__).split('.')[0] # exp_xxx
mode = "local_docker_test"
snapshot_mode = "all"
plot = False
use_gpu = True

# config.DOCKER_IMAGE = 'tsukuyomi2044/rllab'
if "ec2_cpu" in mode:
    config.AWS_INSTANCE_TYPE = "m4.large"
    config.AWS_SPOT_PRICE = '0.1'
    plot = False
    raise NotImplementedError
elif "ec2_gpu" in mode:
    config.AWS_INSTANCE_TYPE = "g2.2xlarge"
    config.AWS_SPOT_PRICE = '1.0'
    plot = False

# different training params ------------------------------------------
from rllab.misc.instrument import VariantGenerator, variant
class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1,101]

    @variant
    def bonus_coeff(self):
        return [0., 1.]

    @variant
    def dim_key(self):
        return [64, 128]

    @variant
    def game(self):
        return ["breakout", "freeway"]

variants = VG().variants()

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
    if ("ec2" in mode) and (len(exp_name) > 64):
        print("Should not use experiment name with length %d > 64.\nThe experiment name is %s.\n Exit now."%(len(exp_name),exp_name))
        sys.exit(1)

    # parameters -----------------------------------------------
    # Experiment Parameters
    # ----------------------
    steps_per_epoch = 300
    epochs = 2
    steps_per_test = 300

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
    update_rule = 'rmsprop'
    batch_accumulator = 'mean'
    learning_rate = .0002
    discount = .95
    rms_decay = .99 # (rho)
    rms_epsilon = 1e-6
    momentum = 0
    clip_delta = 0
    epsilon_start = 1.0
    epsilon_min = .1
    epsilon_decay = 1000000
    phi_length = 4
    update_frequency = 1
    replay_memory_size = 1000000
    batch_size = 32
    network_type = "nips_dnn"
    freeze_interval = 1
    replay_start_size = 100
    resize_method = 'crop'
    resized_width = 84
    resized_height = 84
    death_ends_episode = False
    max_start_nullops = 0
    cudnn_deterministic = False # setting True can result in error
    use_double = False
    clip_reward = True
    agent_unpicklable_list = ["data_set","test_data_set"]
    seed = v["seed"]
    recording = False

    if cudnn_deterministic:
        theano.config.dnn.conv.algo_bwd = 'deterministic'
    freeze_interval = freeze_interval // update_frequency # see launcher.py for a reason

    # setup ALE (not stubbed) --------------------------------------------
    full_rom_path = os.path.join(base_rom_path, rom)
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


    img_preprocessor = ImageVectorizePreprocessor(
        n_chanllel=phi_length,
        width=resized_width,
        height=resized_width,
    )
    hash_list = [
        SimHash(
            item_dim=img_preprocessor.get_output_dim(), # get around stub
            dim_key=v["dim_key"],
            bucket_sizes=None,
        )
    ]
    bonus_evaluator = ALEHashingBonusEvaluator(
        state_dim=img_preprocessor.get_output_dim(),
        img_preprocessor=img_preprocessor,
        num_actions=min_action_set_length,
        hash_list=hash_list,
        count_mode="s",
        bonus_mode="s_next",
        bonus_coeff=v["bonus_coeff"],
        state_bonus_mode="1/n_s",
        state_action_bonus_mode="log(n_s)/n_sa",
    )
    q_network = DeepQLearner(
        input_width=resized_width,
        input_height=resized_height,
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
        update_rule=update_rule,
        batch_accumulator=batch_accumulator,
        input_scale=255.0,
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
logs = []
logs += ["branch: %s" %(repo.active_branch.name)]
logs += ["commit SHA: %s"%(repo.head.object.hexsha)]
logs += exp_names

cur_script_name = __file__
log_file_name = cur_script_name.split('.py')[0] + '.log'
with open(log_file_name,'w') as f:
    for message in logs:
        f.write(message + "\n")

# make the current script read-only to avoid accidental changes after ec2 runs
if "local" not in mode:
    os.system("chmod 444 %s"%(__file__))
