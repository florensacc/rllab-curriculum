from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import cv2
from random import randint
from random import uniform
import numpy as np

blue = (255, 0, 0)


# Environment where agent must scale to appropriate size

class VariableCropEnv(Env):

    def __init__(self, x = 200, y = 300, radius = 15, max_init_distance = 10, correctness = 2, square_to_circle = 1,
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
        self.side = 227

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(self.side, self.side, 3))

    @property
    def action_space(self):
        # print([1.0 * self.max_action, 1.0 * self.max_action, self.max_scale_action])
        return Box(low=np.array([-1.0, -1.0, 1.0 / self.max_scale_action]),
                   high=np.array([1.0, 1.0, self.max_scale_action]))

    def reset(self):
        """
        Radius of the circle is set uniformly between min radius and max radius
        Box coordinates are initialized to be close to the circle
        Box is initialized to fixed size.
        :return:
        """

        #creates a blank image
        self.img = np.zeros((self.y, self.x, 3)) # I think this is correct
        self.add_noise = True
        if self.add_noise:
            cv2.randn(self.img, (240, 240, 240), (100, 100, 100))
        else:
            self.img[:] = (255, 255, 255)

        self.circle_scale = uniform(self.min_scale, self.max_scale)
        self.circle_scaled_radius = int(self.circle_scale * self.radius)
        init_distance = int(self.circle_scaled_radius + self.max_init_distance + self.square_to_circle)
        self.circle_center = randint(init_distance, self.x - init_distance), \
                             randint(init_distance, self.y - init_distance)
        cv2.circle(self.img, self.circle_center, self.circle_scaled_radius, blue, thickness = -1)

        self.box_x = self.circle_center[0] + randint(self.max_init_distance * -1, self.max_init_distance) - self.half_side
        self.box_y = self.circle_center[1] + randint(self.max_init_distance * -1, self.max_init_distance) - self.half_side
        self.box_scale = 1
        self.box_scaled_side = self.side
        self.check_within_bounds()
        return self.get_observation()

    def check_within_bounds(self):
        self.box_x = max(self.box_x, 0)
        self.box_x = int(min(self.box_x, self.x - self.box_scaled_side))
        self.box_y = max(self.box_y, 0)
        self.box_y = int(min(self.box_y, self.y - self.box_scaled_side))

    def get_observation(self):
        # TODO: may need to reshape
        orig = np.copy(self.img[self.box_y: self.box_y + self.box_scaled_side, self.box_x: self.box_x + self.box_scaled_side , :])
        return cv2.resize(orig, (self.side, self.side))

    def step(self, action):
        self.box_x = self.box_x + action[0] * self.max_action
        self.box_y = self.box_y + action[1] * self.max_action
        self.box_scale = self.box_scale * action[2]
        self.box_scaled_side = int(self.box_scale * self.side)
        self.check_within_bounds()
        dist_x = abs(self.circle_center[0] - (self.box_x + self.box_scaled_side / 2))
        dist_y = abs(self.circle_center[1] - (self.box_y + self.box_scaled_side / 2))
        scale_diff = abs((self.box_scale / self.circle_scale) - 1)
        # First component is distance between 2 centers and second component is scale penalty
        reward = - (dist_x ** 2 + dist_y ** 2) ** 0.5 - scale_diff * 2
        done = dist_x <= self.correctness and dist_y <= self.correctness and scale_diff < 0.1

        return Step(observation=self.get_observation(), reward=reward, done=done)

    def render(self, close = True): # not sure what close does, very hacky
        copy_img = np.copy(self.img)
        self.check_within_bounds()
        cv2.rectangle(copy_img, (self.box_x, self.box_y), \
                      (self.box_x + self.box_scaled_side, self.box_y + self.box_scaled_side), (0,0,255), 2)

        crop = self.get_observation() # what the network sees

        cv2.imshow("image_crop", crop)
        cv2.imshow('image', copy_img)
        cv2.waitKey(500)
        print('circle center:' + str(self.circle_center))
        print(self.box_x, self.box_x + self.side)
        print(self.box_y, self.box_y + self.side)

