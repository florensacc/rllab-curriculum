from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import cv2
from random import randint
import numpy as np

blue = (255, 0, 0)


# box_coordinates is upper left hand corner of bounding box, which is initalized randomly
# correctness condition should be an option

class ImageCropEnv(Env):

    def __init__(self, x = 200, y = 300, radius = 15, max_init_distance = 10, correctness = 4, square_to_circle = 0, max_action = 5):
        """

        :param x: x pixels
        :param y: y pixels
        :param radius: radius of circle
        :param max_init_distance: how far the center of the square can be from the circle
        :param correctness: pixel correctness definition
        :param square_to_circle: pixel difference betweeen square size and circle size
        :param max_action: how many pixels the agent can move in a given step
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.max_init_distance = max_init_distance
        self.square_to_circle = square_to_circle
        # half-side should be radius size, just need to make sure max_init_distance < radius * sqrt(2)
        self.half_side = (max_init_distance + radius + square_to_circle)
        self.side = self.half_side * 2
        self.correctness = correctness
        self.max_action = max_action

    @property
    def observation_space(self):
        return Box(low=0, high=255, shape=(self.side, self.side, 3))

    @property
    def action_space(self):
        return Box(low=-1 * self.max_action, high=self.max_action, shape=(2,))

    def reset(self):
        # self.img = np.zeros((self.x, self.y, 3))
        self.img = np.zeros((self.y, self.x, 3)) # I think this is correct
        self.img[:] = (255, 255, 255)

        # deterministic
        # self.circle_center = 70, 100 # make sure circle is correctly oriented (may need to flip)

        self.square_to_circle = 2
        self.circle_center = randint(self.radius + self.max_init_distance + self.square_to_circle, self.x - self.radius - self.max_init_distance - self.square_to_circle), \
                            randint(self.radius + self.max_init_distance + self.square_to_circle, self.y - self.radius - self.max_init_distance - self.square_to_circle)
        cv2.circle(self.img, self.circle_center, self.radius, blue, thickness = -1)

        # deterministic initialization of box
        # self.box_x = self.circle_center[0] + self.max_init_distance - self.half_side
        # self.box_y = self.circle_center[1] + self.max_init_distance - self.half_side

        # random initialization of box
        self.box_x = self.circle_center[0] + randint(self.max_init_distance * -1, self.max_init_distance) - self.half_side
        self.box_y = self.circle_center[1] + randint(self.max_init_distance * -1, self.max_init_distance) - self.half_side
        self.check_within_bounds()
        return self.get_observation()

    # if center of box is too close to the edge, move it further from the edge
    # this might need a penalty term?
    def check_within_bounds(self):
        self.box_x = max(self.box_x, 0)
        self.box_x = int(min(self.box_x, self.x - self.side))
        self.box_y = max(self.box_y, 0)
        self.box_y = int(min(self.box_y, self.y - self.side))

    def get_observation(self):
        return np.copy(self.img[self.box_y: self.box_y + self.side, self.box_x: self.box_x + self.side , :])
        # return np.copy(self.img[self.box_x: self.box_x + self.side, self.box_y: self.box_y + self.side , :])

    def step(self, action):
        # print(action)
        self.box_x = self.box_x + action[0]
        self.box_y = self.box_y + action[1]
        self.check_within_bounds()
        dist_x = abs(self.circle_center[0] - (self.box_x + self.half_side))
        dist_y = abs(self.circle_center[1] - (self.box_y + self.half_side))
        reward = - (dist_x ** 2 + dist_y ** 2) ** 0.5
        done = dist_x < self.correctness and dist_y < self.correctness

        return Step(observation=self.get_observation(), reward=reward, done=done)

    def render(self, close = True): # not sure what close does, very hacky
        copy_img = np.copy(self.img)
        self.check_within_bounds()
        cv2.rectangle(copy_img, (self.box_x, self.box_y), \
                      (self.box_x + self.side, self.box_y + self.side), (0,255,0), 2)

        # crop = np.copy(self.img[self.box_y: self.box_y + self.side, self.box_x: self.box_x + self.side , :])
        crop = self.get_observation()

        cv2.imshow("image_crop", crop)
        cv2.imshow('image', copy_img)
        cv2.waitKey(500)
        print('circle center:' + str(self.circle_center))
        print(self.box_x, self.box_x + self.side)
        print(self.box_y, self.box_y + self.side)

