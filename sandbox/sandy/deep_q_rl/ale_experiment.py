"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague

"""
from rllab.misc import logger
import numpy as np
import cv2
import time
import sys, os
import atari_py

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 0  # Used to be 8 (in Haoran's version)


class ALEExperiment(object):
    def __init__(self, ale_args, agent, env, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops,
                 length_in_episodes=False, max_episode_length=np.inf,
                 game='', observation_type="image", record_image=True,
                 record_ram=False,
                 record_rgb_image=False,
                 recorded_rgb_image_scale=1.,
                 ):
        self.ale_args = ale_args
        self.env = env
        self.env.set_seed(ale_args["seed"])

        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode
        self.min_action_set = self.env.minimal_action_set
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method
        self.width, self.height = self.env.ale_screen_dims

        #self.buffer_length = 2
        #self.buffer_count = 0
        #self.screen_buffer = np.empty((self.buffer_length,
        #                               self.height, self.width),
        #                              dtype=np.uint8)

        self.terminal_lol = False # Most recent episode ended on a loss of life
        self.max_start_nullops = max_start_nullops
        if self.max_start_nullops > 0:
            raise NotImplementedError

        # Whether the lengths (test_length and epoch_length) are specified in
        # episodes. This is mainly for testing
        self.length_in_episodes = length_in_episodes
        self.max_episode_length = max_episode_length

        # allows using RAM state for state counting or even q-learning
        assert observation_type in ["image","ram"]
        if observation_type == "image":
            assert record_image
        elif observation_type == "ram":
            assert record_ram
        self.observation_type = observation_type
        self.record_image = record_image
        self.record_ram = record_ram
        if record_ram:
            self.ram_state = np.zeros(self.env.ale_ram_size, dtype=np.uint8)
        self.record_rgb_image = record_rgb_image
        self.recorded_rgb_image_scale = recorded_rgb_image_scale

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.agent.start_epoch(epoch,phase="Train")
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch,phase="Train")

            if self.test_length > 0:
                self.agent.start_testing(epoch)
                self.agent.start_epoch(epoch,phase="Test")
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_epoch(epoch,phase="Test")
                self.agent.finish_testing(epoch)
            logger.dump_tabular(with_prefix=False)
        self.agent.cleanup()

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        phase = "Test" if testing else "Train"
        self.terminal_lol = False # Make sure each epoch starts with a reset.
        steps_left = num_steps
        start_time = time.clock()
        episode_count = 0
        episode_reward_list = []
        episode_length_list = []
        while steps_left > 0:
            max_steps = np.amin([steps_left, self.max_episode_length])
            _, episode_length, episode_reward = self.run_episode(max_steps, testing)
            episode_reward_list.append(episode_reward)
            episode_length_list.append(episode_length)
            total_time = time.clock() - start_time
            episode_count += 1
            logger.log("""
                {phase} epoch: {epoch_count}, steps left: {steps_left}, total time: {total_time},
                episode: {episode_count}, episode length: {episode_length}, episode reward: {episode_reward},
                """.format(
                phase=phase,
                epoch_count=epoch,
                steps_left=steps_left,
                total_time="%.0f secs"%(total_time),
                episode_count=episode_count,
                episode_length=episode_length,
                episode_reward=episode_reward,
                ))
            steps_left -= episode_length

        # logging
        if phase == "Train":
            logger.record_tabular("Epoch",epoch)
        logger.record_tabular("%sEpochTime"%(phase),"%.0f"%(total_time))
        logger.record_tabular("%sAverageReturn"%(phase), np.average(episode_reward_list))
        logger.record_tabular("%sStdReturn"%(phase), np.std(episode_reward_list))
        logger.record_tabular("%sMedianReturn"%(phase), np.median(episode_reward_list))
        logger.record_tabular("%sMaxReturn"%(phase), np.amax(episode_reward_list))
        logger.record_tabular("%sMinReturn"%(phase), np.amin(episode_reward_list))

        logger.record_tabular("%sAverageEpisodeLength"%(phase), np.average(episode_length_list))
        logger.record_tabular("%sStdEpisodeLength"%(phase), np.std(episode_length_list))
        logger.record_tabular("%sMedianEpisodeLength"%(phase), np.median(episode_length_list))
        logger.record_tabular("%sMaxEpisodeLength"%(phase), np.amax(episode_length_list))
        logger.record_tabular("%sMinEpisodeLength"%(phase), np.amin(episode_length_list))

        # save iteration parameters
        logger.log("Saving iteration parameters...")
        params = dict(
            epoch=epoch,
            agent=self.agent,
            env=self.env,
        )
        logger.save_itr_params(epoch,params)

    def _init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""
        if not self.terminal_lol or self.env.is_terminal:
            self.env.reset()

    def _step(self, action):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        self.env.step(action)

        env_info = {}
        if self.record_ram:
            env_info["ram_state"] = self.env.get_ram()
        if self.record_rgb_image:
            rgb_img = self.env.get_screen_rgb()
            scale = self.recorded_rgb_image_scale
            if abs(scale-1.0) > 1e-4:
                rgb_img = cv2.resize(rgb_img, dsize=(0,0),fx=scale,fy=scale)
            env_info["rgb_image"] = rgb_img

        return self.env.reward, env_info

    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        self._init_episode()

        start_lives = self.env.get_lives()

        action = self.agent.start_episode(self.get_observation())
        num_steps = 0
        total_reward = 0
        while True:
            reward, env_info = self._step(action)
            total_reward += reward
            self.terminal_lol = (self.death_ends_episode and not testing and
                                 self.env.get_lives() < start_lives)
            terminal = self.env._is_terminal or self.terminal_lol
            num_steps += 1

            if terminal or num_steps >= max_steps and not self.length_in_episodes:
                self.agent.end_episode(reward, terminal, env_info)
                break

            action = self.agent.step(reward, self.get_observation(),
                env_info)

        # if the lengths are in episodes, this episode counts as 1 "step"
        if self.length_in_episodes:
            return terminal, 1, total_reward
        else:
            return terminal, num_steps, total_reward


    def get_observation(self):
        # Return the most recent observation, *scaled* to be from [0,1] (or [-1,1])
        return self.env.last_state
