from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import cv2
from random import randint
from random import uniform
from examples.process_alov_data import ProcessALOV
import numpy as np
from rllab.misc import logger

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


# Environment where agent must identify new bounding box
#TODO: 1. max_action and correctness should be proprotional to the size of the image and/or existing bounding box
#2. constructor should take in everything in processs alov, individual functions just get outputs

class SingleVideoEnv(Env):

    def __init__(self, image_folder, image_name, annotation_file_name, side = 200,
                 max_action = 10, max_scale = 2, min_scale = 0.5, correctness = 25, fixed_sample = False):
        """
        :param max_action: how many pixels the agent can move in a given step
        :param max_scale_action maximum change in scale in a given step (ranges from 1 / max_scale_action to max_scale_action)
        :param max_scale maximum scale of the circle
        :param min_scale minimum scale of the circle
        """
        processor = ProcessALOV()
        self.examples = processor.get_training_data(image_folder, image_name, annotation_file_name)
        self.training_examples = len(self.examples)
        self.max_action = max_action
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.side = side
        self.correctness = correctness
        self.fixed_sample = fixed_sample

    @property
    def observation_space(self):
        """
        :return: box where the first three color channels are for the first crop and where the second three color
        channels are for the second crop
        """
        return Box(low=0, high=255, shape=(self.side, self.side, 6))

    @property
    def action_space(self):
        """
        :return: x_change, y_change, x_scale, y_scale
        """
        return Box(low=np.array([-1.0, -1.0,
                                self.min_scale, self.min_scale]),
                   high=np.array([1.0, 1.0,
                                 self.max_scale, self.max_scale]))

    def reset(self):
        """
        Radius of the circle is set uniformly between min radius and max radius
        Box coordinates are initialized to be close to the circle
        Box is initialized to fixed size.
        :return:
        """
        i = randint(0, self.training_examples - 1)
        if self.fixed_sample:
            sample = self.examples[0]  # fixed sample for easier training
        else:
            sample = self.examples[i]

        self.old_img, self.tl, self.br = sample[0]
        self.new_img, self.new_tl, self.new_br = sample[1]
        self.x_max, self.y_max, _ = self.old_img.shape
        self.old_crop = cv2.resize(self.old_img[self.tl[1]:self.br[1], self.tl[0]:self.br[0], :], (self.side, self.side))
        self.x_center = int((self.br[1] + self.tl[1]) / 2)
        self.y_center = int((self.br[0] + self.tl[0]) / 2)

        # should rename variable half-width and half-height
        self.half_width = (self.br[1] - self.tl[1]) / 1.5 # TODO: this is how much the network sees, should check
        self.half_height = (self.br[0] - self.tl[0]) / 1.5
        self.max_action = (self.half_width + self.half_height) / 4 # TODO:check

        self.target_x_center = (self.new_br[1] + self.new_tl[1]) / 2
        self.target_y_center = (self.new_br[0] + self.new_tl[0]) / 2
        self.target_half_width = (self.new_br[1] - self.new_tl[1]) / 2
        self.target_half_height = (self.new_br[0] - self.new_tl[0]) / 2

        self.check_within_bounds()
        return self.get_observation()

    def check_within_bounds(self):
        self.half_width = int(self.half_width)
        self.half_height = int(self.half_height)
        if self.x_center - self.half_width < 0:
            self.x_center = self.half_width
        if self.x_center + self.half_width > self.x_max:
            self.x_center = self.x_max - self.half_width
        if self.y_center - self.half_height < 0:
            self.y_center = self.half_height
        if self.y_center + self.half_height > self.y_max:
            self.y_center = self.y_max - self.half_height

    def get_observation(self):
        # old is box around old image
        current = np.copy(self.new_img[self.x_center - self.half_width:self.x_center + self.half_width,
                          self.y_center - self.half_height:self.y_center + self.half_height, :])
        new_crop = cv2.resize(current, (self.side, self.side))
        # print(new_crop.shape)
        # print(self.old_crop.shape)
        return np.concatenate((self.old_crop, new_crop), axis = 2)

    def corner_difference_reward(self):
        self.top_x = self.x_center - self.half_width
        self.top_y = self.y_center - self.half_height
        self.br_x = self.x_center + self.half_width
        self.br_y = self.y_center + self.half_height

        return -1 * (abs(self.top_x - self.new_tl[1]) + abs(self.top_y -self.new_tl[0]) +
                     abs(self.br_x - self.new_br[1]) + abs(self.br_y - self.new_br[0]))

    def step(self, action):
        # make sure that each side is scaled appropriately
        self.x_center = self.x_center + action[0] * self.max_action
        self.y_center= self.y_center + action[1] * self.max_action
        self.half_width = self.half_width * action[2]
        self.half_height = self.half_height * action[3]
        self.check_within_bounds()
        dist_x = abs(self.target_x_center - self.x_center)
        dist_y = abs(self.target_y_center - self.y_center)
        # penalize x different in scale and y difference in scale
        scale_diff_x = abs((self.half_width / self.target_half_width) - 1)
        scale_diff_y = abs((self.half_height / self.target_half_height) - 1)
        scale_diff = scale_diff_x + scale_diff_y
        # First component is distance between 2 centers and second component is scale penalty
        # reward = - (dist_x ** 2 + dist_y ** 2) ** 0.5 - scale_diff
        reward = self.corner_difference_reward()
        done = dist_x <= self.correctness and dist_y <= self.correctness and scale_diff < 0.3

        return Step(observation=self.get_observation(), reward=reward, done=done, dist_x=dist_x, dist_y = dist_y,
                    scale_diff_x = scale_diff_x, scale_diff_y = scale_diff_y)

    def render(self, close = True): # not sure what close does, very hacky
        copy_img = np.copy(self.new_img)
        self.check_within_bounds()
        cv2.rectangle(copy_img, (int(self.y_center - self.half_height), int(self.x_center - self.half_width)),
                      (int(self.y_center + self.half_height), int(self.x_center + self.half_width)), red, 2)

        observation = self.get_observation()
        old_crop =  observation[:, :, :3]  # what the network sees
        crop = observation[:,:,3:] # what the network sees

        cv2.imshow("old_crop", old_crop)
        cv2.imshow("image_crop", crop)
        cv2.imshow('image', copy_img)
        cv2.waitKey(0)


        print("target x: %d" % self.target_x_center)
        print("target y: %d" % self.target_y_center)
        print("target x scale: %d" % self.target_half_width)
        print("target y scale: %d" % self.target_half_height)
        print("current x: %d" % self.x_center)
        print("current y: %d" % self.y_center)
        print("current x scale: %d" % self.half_width)
        print("current y scale: %d" % self.half_height)

    # @ override?
    def log_diagnostics(self, paths):
        dist_x = [path["env_infos"]["dist_x"] for path in paths]
        logger.record_tabular('dist_x', np.mean([np.min(d) for d in dist_x]))
        dist_y = [path["env_infos"]["dist_y"] for path in paths]
        logger.record_tabular('dist_y', np.mean([np.min(d) for d in dist_y]))
        scale_diff_x = [path["env_infos"]["scale_diff_x"] for path in paths]
        logger.record_tabular('scale_diff_x', np.mean([np.min(d) for d in scale_diff_x]))
        scale_diff_y = [path["env_infos"]["scale_diff_y"] for path in paths]
        logger.record_tabular('scale_diff_y', np.mean([np.min(d) for d in scale_diff_y]))



