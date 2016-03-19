# Imports
import numpy as np
import random
import pygame
from rllab.mdp.base import MDP
#
# # Start PyGame
# pygame.init()

# Define Colors
black = [0, 0, 0]
white = [255, 255, 255]
blue = [0, 0, 255]
green = [0, 255, 0]
red = [255, 0, 0]


class Block(pygame.sprite.Sprite):
    # Default color of a block
    color = black
    # Default size of a block
    size = 10

    def __init__(self, locX, locY):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)
        # Create an image of the block, and fill it with a color
        # This could also be an image loaded from the disk
        self.image = pygame.Surface([self.size, self.size])
        self.image.fill(self.color)
        # Set bounds
        self.rect = self.image.get_rect()
        # Set draw location
        self.rect.x = locX
        self.rect.y = locY


def _row_in_mat(row, mat):
    """
    Test if a row vector is present in a matrix
    :param row: row vector
    :param mat: matrix
    :return: whether the row vector is present in the matrix
    """
    return np.any(np.array_equal(row, x) for x in mat)


class SortingGridMDP(MDP):
    """
    Creates a square grid of the specified size (must be an odd number). The agent starts at a random position on the
    grid, and an object lies at the center of the grid. The goal is to pick up the object
    and bring it to a particular destination which is unknown to the agent and randomized per episode. The agent only
    has one shot to attempt to put the object in. Hence the optimal strategy is to pick up the object, decide on a
    target position at random, and head there as soon as possible.

    In principle, a recurrent neural network has the capacity to represent such an optimal policy (the same could be
    said to general hierarchical RL problem. The goal is to investigate whether it is indeed possible to train such
    policies, and, if not, how to improve upon the naive training strategy.
    """

    ACTION_MAP = np.array([
        [0, -1],
        [-1, 0],
        [0, 1],
        [1, 0],
    ])

    def __init__(self,
                 world_height=1,
                 world_width=11,
                 object_init_positions=None,
                 agent_init_positions=None,
                 object_final_positions=None):
        if object_init_positions is None:
            object_init_positions = [[0, 0]]
        if agent_init_positions is None:
            agent_init_positions = [[world_height / 2, world_width / 2]]
        if object_final_positions is None:
            # default to the four corners of the grid world
            object_final_positions = [
                [0, world_width - 1],
                [world_height - 1, world_width - 1],
                [world_height - 1, 0],
                [0, 0]
            ]
        self._object_init_positions = np.array(object_init_positions)
        self._agent_init_positions = np.array(agent_init_positions)
        self._object_final_positions = np.array(object_final_positions)
        self._world_height = world_height
        self._world_width = world_width
        self._agent_pos = None
        self._object_pos = None
        self._target_pos = None
        self._object_picked_up = False

    def reset(self):
        self._agent_pos = random.choice(self._agent_init_positions)
        self._object_pos = random.choice(self._object_init_positions)
        self._target_pos = random.choice(self._object_final_positions)
        self._object_picked_up = False

    @property
    def action_dim(self):
        return len(self.ACTION_MAP)

    @property
    def action_dtype(self):
        return 'uint8'

    @property
    def observation_shape(self):
        """
        The observation should be a (height * width * 3) binary vector, indicating one of 3 states for each grid:
        empty,
        :return:
        """

    def step(self, action):
        self._agent_pos = np.clip(
            self._agent_pos + np.array(action),
            [0, 0],
            [self._world_height - 1, self._world_width - 1]
        )
        reward = 0
        done = False
        if not self._object_picked_up:
            if np.array_equal(self._agent_pos, self._object_pos):
                self._object_picked_up = True
                # set to some invalid value, because this property should not be used anymore
                self._object_pos = np.array([np.nan, np.nan])
        elif _row_in_mat(self._agent_pos, self._object_final_positions):
            done = True
            if np.array_equal(self._agent_pos, self._target_pos):
                reward = 1
        return self._get_current_obs(), reward, done

    def _get_current_obs(self):
        pass






# Set and Display Screen
# sizeX = 500
# sizeY = 500
# size = [sizeX, sizeY]
# screen = pygame.display.set_mode(size)
#
# # Set Screen's Title
# pygame.display.set_caption("World Grid")
#
# # Grid Settings
# width = 20  # X size of grid location
# height = 20  # Y size of grid location
#
# # Make Grid Array
# grid = []
# for row in range(int(sizeX / width)):
#     grid.append([])
#     for column in range(int(sizeX / height)):
#         grid[row].append(0)
#
# # Test Grid Value
# grid[1][5] = 1
#
# # Sentinel for Game Loop
# done = False
#
# # Game Timer
# clock = pygame.time.Clock()
#
# # Main Game Loop
# while done == False:
#     # Limit FPS of Game Loop
#     clock.tick(25)
#
#     # Check for Events
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             done = True
#         elif event.type == pygame.MOUSEBUTTONDOWN:
#             pos = pygame.mouse.get_pos()
#             gridLocX = int(pos[0] / width)
#             gridLocY = int(pos[1] / height)
#             grid[gridLocY][gridLocX] = 1
#
#     # Clear the Screen
#     screen.fill(black)
#
#     # Draw Grid
#     for column in range(int(sizeX / width)):
#         for row in range(int(sizeY / height)):
#             if grid[row][column] == 1:
#                 color = green
#             else:
#                 color = white
#             pygame.draw.rect(screen, color, [column * width, row * height, width, height])
#
#     # Update Display
#     pygame.display.flip()
#
# # Exit Program
# pygame.quit()
