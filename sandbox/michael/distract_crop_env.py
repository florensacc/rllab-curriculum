from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import cv2
from random import randint
from random import uniform
import numpy as np

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)


# Environment where agent must scale to appropriate size and ignore distractor object

class DistractCropEnv(Env):

    def __init__(self, x = 200, y = 300, radius = 15, max_init_distance = 10, correctness = 10, square_to_circle = 1,
                 max_action = 10, max_scale_action = 1.15, max_scale = 1.3, min_scale = 0.7, add_noise = True):
        """

        :param x: x pixels
        :param y: y pixels
        :param radius: radius of circle
        :param max_init_distance: maximum distance from center of the square to the circle
        :param correctness: pixel correctness definition (agent is done when within correctness pixels)
        :param square_to_circle: pixel difference betweeen square size and circle size
        (positive means bounding box is larger and should make more sense?)
        :param max_action: how many pixels the agent can move in a given step
        :param max_scale_action maximum change in scale in a given step (ranges from 1 / max_scale_action to max_scale_action)
        :param max_scale maximum scale of the circle
        :param min_scale minimum scale of the circle
        :param add_noise whether or not noise is added to the background
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.max_init_distance = max_init_distance
        self.square_to_circle = square_to_circle
        # half-side should be radius size, just need to make sure max_init_distance * sqrt(2) < radius
        if self.max_init_distance > self.radius / 1.5: #want to make sure circle is in radius
            print("Max init distance too far!")
            self.max_init_distance = int(self.radius / 1.5)
        self.half_side = radius + square_to_circle
        self.side = self.half_side * 2
        self.correctness = correctness
        self.max_action = max_action
        self.max_scale_action = max_scale_action
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.add_noise = add_noise

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
        return Box(low=np.array([-1.0 * self.max_action, -1.0 * self.max_action,
                                 1.0 / self.max_scale_action, 1.0 / self.max_scale_action]),
                   high=np.array([1.0 * self.max_action, 1.0 * self.max_action,
                                  self.max_scale_action, self.max_scale_action]))

    def reset(self):
        """
        Radius of the circle is set uniformly between min radius and max radius
        Box coordinates are initialized to be close to the circle
        Box is initialized to fixed size.
        :return:
        """

        #creates a blank image
        self.img = np.zeros((self.y, self.x, 3)) # I think this is correct
        if self.add_noise:
            cv2.randn(self.img, (240, 240, 240), (100, 100, 100))
        else:
            self.img[:] = (255, 255, 255)

        self.circle_scale = uniform(self.min_scale, self.max_scale)
        self.circle_scaled_radius = int(self.circle_scale * self.radius)
        init_distance = int(self.circle_scaled_radius + self.max_init_distance + self.square_to_circle)
        self.circle_center = randint(init_distance, self.x - init_distance), \
                             randint(init_distance, self.y - init_distance)

        self.box_x = self.circle_center[0] + randint(self.max_init_distance * -1, self.max_init_distance) \
                     - self.half_side
        self.box_y = self.circle_center[1] + randint(self.max_init_distance * -1, self.max_init_distance) \
                     - self.half_side
        self.box_scale_x, self.box_scale_y = 1, 1
        self.box_scaled_side_x, self.box_scaled_side_y = self.side, self.side
        self.check_within_bounds()

        self.distractor_center = randint(self.box_x - self.max_init_distance, self.box_x + self.max_init_distance), \
                                randint(self.box_y - self.max_init_distance, self.box_y + self.max_init_distance)
        target = randint(1, 2)
        if target == 1:
            cv2.circle(self.img, self.distractor_center, self.circle_scaled_radius, blue, thickness=-1)
            cv2.circle(self.img, self.circle_center, self.circle_scaled_radius, green, thickness=-1)
        else:
            # print("targetting blue")
            cv2.circle(self.img, self.distractor_center, self.circle_scaled_radius, green, thickness=-1)
            cv2.circle(self.img, self.circle_center, self.circle_scaled_radius, blue, thickness=-1)

        return self.get_observation()

    def check_within_bounds(self):
        self.box_x = max(self.box_x, 0)
        self.box_x = int(min(self.box_x, self.x - self.box_scaled_side_x))
        self.box_y = max(self.box_y, 0)
        self.box_y = int(min(self.box_y, self.y - self.box_scaled_side_y))

    def get_observation(self):
        # old is box around old image
        old = np.copy(self.img[self.circle_center[1] - self.circle_scaled_radius: self.circle_center[1] + self.circle_scaled_radius,
                      self.circle_center[0] - self.circle_scaled_radius: self.circle_center[0] + self.circle_scaled_radius, :])
        old_scaled = cv2.resize(old, (self.side, self.side))
        current = np.copy(self.img[self.box_y: self.box_y + self.box_scaled_side_x, self.box_x: self.box_x + self.box_scaled_side_y , :])
        current_scaled = cv2.resize(current, (self.side, self.side))
        return np.concatenate((old_scaled, current_scaled), axis = 2)

    def step(self, action):
        # make sure that each side is scaled appropriately
        self.box_x = self.box_x + action[0]
        self.box_y = self.box_y + action[1]
        self.box_scale_x = self.box_scale_x * action[2]
        self.box_scale_y = self.box_scale_y * action[3]
        self.box_scaled_side_x = int(self.box_scale_x * self.side)
        self.box_scaled_side_y = int(self.box_scale_y * self.side)
        self.check_within_bounds()
        dist_x = abs(self.circle_center[0] - (self.box_x + self.box_scaled_side_x / 2))
        dist_y = abs(self.circle_center[1] - (self.box_y + self.box_scaled_side_y / 2))
        # penalize x different in scale and y difference in scale
        scale_diff = abs((self.box_scale_x / self.circle_scale) - 1) + abs((self.box_scale_y / self.circle_scale) - 1)
        # First component is distance between 2 centers and second component is scale penalty
        reward = - (dist_x ** 2 + dist_y ** 2) ** 0.5 - scale_diff * 5 # heavier penalty on scale
        done = dist_x <= self.correctness and dist_y <= self.correctness and scale_diff < 0.3

        return Step(observation=self.get_observation(), reward=reward, done=done)

    def render(self, close = True): # not sure what close does, very hacky
        copy_img = np.copy(self.img)
        self.check_within_bounds()
        cv2.rectangle(copy_img, (self.box_x, self.box_y), \
                      (self.box_x + self.box_scaled_side_x, self.box_y + self.box_scaled_side_y), red, 2)

        observation = self.get_observation()
        old_crop =  cv2.resize(observation[:, :, :3], (80, 80))  # what the network sees
        crop = cv2.resize(observation[:,:,3:], (80, 80)) # what the network sees

        cv2.imshow("old_crop", old_crop)
        cv2.imshow("image_crop", crop)
        cv2.imshow('image', copy_img)
        cv2.waitKey(400)
        print('circle center:' + str(self.circle_center))
        print(self.box_x, self.box_x + self.side)
        print(self.box_y, self.box_y + self.side)

