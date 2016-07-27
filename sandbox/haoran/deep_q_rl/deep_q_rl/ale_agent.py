"""
The NeuralAgent class wraps a deep Q-network for training and testing
in the Arcade learning environment.

Author: Nathan Sprague

"""

import os
import cPickle
import time
from rllab.misc import logger

import numpy as np

import ale_data_set

import sys
sys.setrecursionlimit(10000)

class NeuralAgent(object):

    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, experiment_directory,
                 replay_start_size, update_frequency, rng, clip_reward, bonus_evaluator=None,
                 recording=True):

        self.results_file = self.learning_file = None
        self.best_epoch_reward = None

        self.network = q_network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.rng = rng

        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height

        self.recording = recording

        self.clip_reward = clip_reward

        self.exp_dir = experiment_directory
        if self.recording:
            try:
                os.stat(self.exp_dir)
            except OSError:
                os.makedirs(self.exp_dir)
            self.record_parameters()


        self.num_actions = self.network.num_actions


        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             rng=rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  rng=rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None

        self.bonus_evaluator = bonus_evaluator

    def _open_results_file(self):
        if not self.recording:
            return
        logger.log("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,best_reward,mean_q\n')
        self.results_file.flush()

    def _open_learning_file(self):
        if not self.recording:
            return
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        if not self.recording:
            return
        out = "{},{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                        self.total_reward / max(1.0, float(num_episodes)),
                                        self.best_epoch_reward,
                                        holdout_sum)
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        if not self.recording:
            return
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action


    def record_parameters(self):
        if not self.recording:
            return

        import subprocess

        parameters_filename = os.path.join(self.exp_dir, 'parameters.txt')
        with open(parameters_filename, 'w') as parameters_file:
            # write the commit we are at
            # gitlog = subprocess.check_output('git log -n 1 --oneline'.split()).strip()
            # parameters_file.write('Last commit: %s\n' % gitlog)

            for variable in sorted('epsilon_start epsilon_min epsilon_decay phi_length replay_memory_size   replay_start_size update_frequency'.split()):
                parameters_file.write('%s: %s\n' % (variable, getattr(self, variable)))
                logger.log('%s: %s' % (variable, getattr(self, variable)))

        gitdiff = subprocess.check_output('git diff'.split()).strip()
        if gitdiff:
            diff_filename = os.path.join(self.exp_dir, 'difftogit.txt')
            with open(diff_filename, 'w') as diff_file:
                diff_file.write(gitdiff)
                diff_file.write('\n')


    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1

        #TESTING---------------------------
        if self.clip_reward:
            reward = np.clip(reward, -1, 1)
        if self.testing:
            self.episode_reward += reward
            action = self._choose_action(self.test_data_set, .05,
                                         observation, reward)

        #NOT TESTING---------------------------
        else:

            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)

                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             reward)

                if self.step_counter % self.update_frequency == 0:
                    loss = self._do_training()
                    self.batch_counter += 1
                    self.loss_averages.append(loss)

            else: # Still gathering initial random data...
                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             reward)

        self.last_action = action
        self.last_img = observation

        return action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
            if self.bonus_evaluator is not None:
                states = np.asarray([phi])
                actions = np.asarray([action])
                self.bonus_evaluator.update(states,actions)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        states, actions, rewards, next_states, terminals = \
                                self.data_set.random_batch(
                                    self.network.batch_size)

        if self.bonus_evaluator is not None:
            bonus_rewards = self.bonus_evaluator.evaluate(states,actions,next_states)
            rewards += bonus_rewards.reshape((len(bonus_rewards),1))
        loss =  self.network.train(states, actions, rewards, next_states, terminals)
        return loss


    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward

            # keep track of the best reward for the epoch
            if self.best_epoch_reward is None or self.episode_reward > self.best_epoch_reward:
                self.best_epoch_reward = self.episode_reward
        else:

            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True)

            logger.log("steps/second: {:.2f}".format(\
                            self.step_counter/total_time))

            if self.batch_counter > 0:
                self._update_learning_file()
                logger.log("average loss: {:.4f}".format(\
                                np.mean(self.loss_averages)))


    def finish_epoch(self, epoch):
        if not self.recording:
            return
        net_file = open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'w')
        cPickle.dump(self.network, net_file, -1)
        net_file.close()

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0
        self.best_epoch_reward = None

    def finish_testing(self, epoch):
        self.testing = False
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            self.holdout_data = self.data_set.random_batch(holdout_size)[0]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i, ...]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)


    def cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """

        if self.learning_file:
            self.learning_file.close()
        if self.results_file:
            self.results_file.close()


if __name__ == "__main__":
    pass
