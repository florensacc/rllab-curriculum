#! /usr/bin/env python
"""This script handles reading command line arguments and starting the
training process.  It shouldn't be executed directly; it is used by
run_nips.py or run_nature.py.

"""
import os, sys
import argparse
from rllab.misc import logger
import time
import ale_python_interface
import cPickle
import numpy as np
import theano

import ale_experiment
import ale_agent
import q_network

from sandbox.haoran.hashing.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.haoran.hashing.preprocessor.image_vectorize_preprocessor import ImageVectorizePreprocessor

class Launcher(object):
    def __init__(self,args,defaults,description):
        self.args = args
        self.defaults = defaults
        self.description = description

    def process_args(self,args, defaults, description):
        """
        Handle the command line.

        args     - list of command line arguments (not including executable name)
        defaults - a name space with variables corresponding to each of
                   the required default command line values.
        description - a string to display at the top of the help message.
        """
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('-r', '--rom', dest="rom", default=defaults.ROM,
                            help='ROM to run (default: %(default)s)')
        parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                            default=defaults.EPOCHS,
                            help='Number of training epochs (default: %(default)s)')
        parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                            type=int, default=defaults.STEPS_PER_EPOCH,
                            help='Number of steps per epoch (default: %(default)s)')
        parser.add_argument('-t', '--test-length', dest="steps_per_test",
                            type=int, default=defaults.STEPS_PER_TEST,
                            help='Number of steps per test (default: %(default)s)')
        parser.add_argument('--display-screen', dest="display_screen",
                            action='store_true', default=False,
                            help='Show the game screen.')
        parser.add_argument('--experiment-prefix', dest="experiment_prefix",
                            default=defaults.EXPERIMENT_PREFIX,
                            help='Experiment name prefix '
                            '(default is the name of the game)')
        parser.add_argument('--frame-skip', dest="frame_skip",
                            default=defaults.FRAME_SKIP, type=int,
                            help='Every how many frames to process '
                            '(default: %(default)s)')
        parser.add_argument('--repeat-action-probability',
                            dest="repeat_action_probability",
                            default=defaults.REPEAT_ACTION_PROBABILITY, type=float,
                            help=('Probability that action choice will be ' +
                                  'ignored (default: %(default)s)'))

        parser.add_argument('--update-rule', dest="update_rule",
                            type=str, default=defaults.UPDATE_RULE,
                            help=('deepmind_rmsprop|rmsprop|sgd ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--batch-accumulator', dest="batch_accumulator",
                            type=str, default=defaults.BATCH_ACCUMULATOR,
                            help=('sum|mean (default: %(default)s)'))
        parser.add_argument('--learning-rate', dest="learning_rate",
                            type=float, default=defaults.LEARNING_RATE,
                            help='Learning rate (default: %(default)s)')
        parser.add_argument('--rms-decay', dest="rms_decay",
                            type=float, default=defaults.RMS_DECAY,
                            help='Decay rate for rms_prop (default: %(default)s)')
        parser.add_argument('--rms-epsilon', dest="rms_epsilon",
                            type=float, default=defaults.RMS_EPSILON,
                            help='Denominator epsilson for rms_prop ' +
                            '(default: %(default)s)')
        parser.add_argument('--momentum', type=float, default=defaults.MOMENTUM,
                            help=('Momentum term for Nesterov momentum. '+
                                  '(default: %(default)s)'))
        parser.add_argument('--clip-delta', dest="clip_delta", type=float,
                            default=defaults.CLIP_DELTA,
                            help=('Max absolute value for Q-update delta value. ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--clip-reward', dest="clip_reward", type=float,
                            default=defaults.CLIP_REWARD,
                            help=('Max absolute value for Q-update delta value. ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--discount', type=float, default=defaults.DISCOUNT,
                            help='Discount rate')
        parser.add_argument('--epsilon-start', dest="epsilon_start",
                            type=float, default=defaults.EPSILON_START,
                            help=('Starting value for epsilon. ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--epsilon-min', dest="epsilon_min",
                            type=float, default=defaults.EPSILON_MIN,
                            help='Minimum epsilon. (default: %(default)s)')
        parser.add_argument('--epsilon-decay', dest="epsilon_decay",
                            type=float, default=defaults.EPSILON_DECAY,
                            help=('Number of steps to minimum epsilon. ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--phi-length', dest="phi_length",
                            type=int, default=defaults.PHI_LENGTH,
                            help=('Number of recent frames used to represent ' +
                                  'state. (default: %(default)s)'))
        parser.add_argument('--max-history', dest="replay_memory_size",
                            type=int, default=defaults.REPLAY_MEMORY_SIZE,
                            help=('Maximum number of steps stored in replay ' +
                                  'memory. (default: %(default)s)'))
        parser.add_argument('--batch-size', dest="batch_size",
                            type=int, default=defaults.BATCH_SIZE,
                            help='Batch size. (default: %(default)s)')
        parser.add_argument('--network-type', dest="network_type",
                            type=str, default=defaults.NETWORK_TYPE,
                            help=('nips_cuda|nips_dnn|nature_cuda|nature_dnn' +
                                  '|linear (default: %(default)s)'))
        parser.add_argument('--freeze-interval', dest="freeze_interval",
                            type=int, default=defaults.FREEZE_INTERVAL,
                            help=('Interval between target freezes. ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--update-frequency', dest="update_frequency",
                            type=int, default=defaults.UPDATE_FREQUENCY,
                            help=('Number of actions before each SGD update. '+
                                  '(default: %(default)s)'))
        parser.add_argument('--replay-start-size', dest="replay_start_size",
                            type=int, default=defaults.REPLAY_START_SIZE,
                            help=('Number of random steps before training. ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--resize-method', dest="resize_method",
                            type=str, default=defaults.RESIZE_METHOD,
                            help=('crop|scale (default: %(default)s)'))
        parser.add_argument('--nn-file', dest="nn_file", type=str, default=None,
                            help='Pickle file containing trained net.')
        parser.add_argument('--death-ends-episode', dest="death_ends_episode",
                            type=str, default=defaults.DEATH_ENDS_EPISODE,
                            help=('true|false (default: %(default)s)'))
        parser.add_argument('--max-start-nullops', dest="max_start_nullops",
                            type=int, default=defaults.MAX_START_NULLOPS,
                            help=('Maximum number of null-ops at the start ' +
                                  'of games. (default: %(default)s)'))
        parser.add_argument('--deterministic', dest="deterministic",
                            type=bool, default=defaults.DETERMINISTIC,
                            help=('Whether to use deterministic parameters ' +
                                  'for learning. (default: %(default)s)'))
        parser.add_argument('--cudnn_deterministic', dest="cudnn_deterministic",
                            type=bool, default=defaults.CUDNN_DETERMINISTIC,
                            help=('Whether to use deterministic backprop. ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--use_double', dest="use_double",
                            type=bool, default=defaults.USE_DOUBLE,
                            help=('Whether to use Double DQN. ' +
                                  '(default: %(default)s)'))
        parser.add_argument('--experiment-directory', dest="experiment_directory",
            default=defaults.EXPERIMENT_DIRECTORY,
            help=('Specify exact directory where to save output to ' +
                '(default: combination of prefix and game name and current ' +
                'date and parameters)'))
        parser.add_argument('--no-record', dest="recording", default=True,
            action="store_false",
            help=('Do not record anything about the experiment ' +
                '(best games, epoch networks, test results, etc)'))
        parser.add_argument('--record-video', dest="record_video",
            default=False, action="store_true",
            help='Record screen captures')
        parser.add_argument('--episodes', dest="episodes", default=False,
            action="store_true",
            help=('This changes the lengths of training epochs and test ' +
            'epochs to be measured in episodes (games) instead of steps'))
        parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                            help='Name of the tabular log file (in csv).')
        parser.add_argument('--use-bonus', type=bool, default=defaults.USE_BONUS, help='Whether to use the bonus reward.')
        parser.add_argument("--agent-unpicklable-list",type=str,nargs='+',help="A list of attributes of the agent that should not be pickled (to save disk space).",default=defaults.AGENT_UNPICKLABLE_LIST)


        parameters = parser.parse_args(args)
        if parameters.experiment_prefix is None:
            parameters.experiment_prefix = ""

        if parameters.death_ends_episode == 'true':
            parameters.death_ends_episode = True
        elif parameters.death_ends_episode == 'false':
            parameters.death_ends_episode = False
        else:
            raise ValueError("--death-ends-episode must be true or false")

        if parameters.freeze_interval > 0:
            # This addresses an inconsistency between the Nature paper and
            # the Deepmind code.  The paper states that the target network
            # update frequency is "measured in the number of parameter
            # updates".  In the code it is actually measured in the number
            # of action choices.
            parameters.freeze_interval = (parameters.freeze_interval //
                                          parameters.update_frequency)

        return parameters



    def launch(self):
        """
        Execute a complete training run.
        """

        args = self.args
        defaults = self.defaults
        description = self.description

        parameters = self.process_args(args, defaults, description)


        if parameters.rom.endswith('.bin'):
            rom = parameters.rom
        else:
            rom = "%s.bin" % parameters.rom
        full_rom_path = os.path.join(defaults.BASE_ROM_PATH, rom)

        if parameters.deterministic:
            rng = np.random.RandomState(123456)
        else:
            rng = np.random.RandomState()

        if parameters.cudnn_deterministic:
            theano.config.dnn.conv.algo_bwd = 'deterministic'


        if parameters.experiment_directory:
            experiment_directory = parameters.experiment_directory
        else:
            game_name = os.path.splitext(os.path.basename(parameters.rom))[0]
            time_str = time.strftime("%Y-%m-%d-%H-%M")
            experiment_directory = os.path.join(
                parameters.experiment_prefix,
                game_name,
                time_str
            )
        tabular_log_file = os.path.join(experiment_directory, parameters.tabular_log_file)
        logger.add_tabular_output(tabular_log_file)


        ale = ale_python_interface.ALEInterface()
        ale.setInt('random_seed', rng.randint(1000))

        if parameters.display_screen:
            if sys.platform == 'darwin':
                import pygame
                pygame.init()
                ale.setBool('sound', False) # Sound doesn't work on OSX

        ale.setBool('display_screen', parameters.display_screen)
        ale.setFloat('repeat_action_probability',
                     parameters.repeat_action_probability)

        if parameters.record_video:
            video_directory = os.path.join(experiment_directory, 'video')
            if not os.path.isdir(video_directory):
                os.makedirs(video_directory)


            ale.setString('record_screen_dir', video_directory)

            if sys.platform != 'darwin':
                ale.setBool('sound', True)
                ale.setString("record_sound_filename", os.path.join(video_directory, "sound.wav"))
                # "We set fragsize to 64 to ensure proper sound sync"
                # (that's what videoRecordingExample.cpp in ALE says. I don't really know what it means)
                ale.setInt("fragsize", 64)

        ale.loadROM(full_rom_path)

        num_actions = len(ale.getMinimalActionSet())

        if parameters.nn_file is None:
            network = q_network.DeepQLearner(
                input_width=defaults.RESIZED_WIDTH,
                input_height=defaults.RESIZED_HEIGHT,
                num_actions=num_actions,
                num_frames=parameters.phi_length,
                discount=parameters.discount,
                learning_rate=parameters.learning_rate,
                rho=parameters.rms_decay,
                rms_epsilon=parameters.rms_epsilon,
                momentum=parameters.momentum,
                clip_delta=parameters.clip_delta,
                freeze_interval=parameters.freeze_interval,
                use_double=parameters.use_double,
                batch_size=parameters.batch_size,
                network_type=parameters.network_type,
                update_rule=parameters.update_rule,
                batch_accumulator=parameters.batch_accumulator,
                rng=rng,
                input_scale=255.0,
                )
        else:
            handle = open(parameters.nn_file, 'r')
            network = cPickle.load(handle)

        if parameters.use_bonus:
            img_preprocessor = ImageVectorizePreprocessor(
                n_chanllel=parameters.phi_length,
                width=defaults.RESIZED_WIDTH,
                height=defaults.RESIZED_HEIGHT,
            )
            bonus_evaluator = ALEHashingBonusEvaluator(
                state_dim=img_preprocessor.output_dim,
                img_preprocessor=img_preprocessor,
                num_actions=num_actions,
                hash_list=[],
                count_mode="s",
                bonus_mode="s_next",
                bonus_coeff=1.0,
                state_bonus_mode="1/n_s",
                state_action_bonus_mode="log(n_s)/n_sa",
            )
        agent = ale_agent.NeuralAgent(network,
              parameters.epsilon_start,
              parameters.epsilon_min,
              parameters.epsilon_decay,
              parameters.replay_memory_size,
              experiment_directory,
              parameters.replay_start_size,
              parameters.update_frequency,
              rng,
              clip_reward=parameters.clip_reward,
              bonus_evaluator=bonus_evaluator,
              recording=parameters.recording)
        experiment = ale_experiment.ALEExperiment(ale, agent,
              defaults.RESIZED_WIDTH,
              defaults.RESIZED_HEIGHT,
              parameters.resize_method,
              parameters.epochs,
              parameters.steps_per_epoch,
              parameters.steps_per_test,
              parameters.frame_skip,
              parameters.death_ends_episode,
              parameters.max_start_nullops,
              rng,
              length_in_episodes=parameters.episodes)

        experiment.run()



if __name__ == '__main__':
    pass
