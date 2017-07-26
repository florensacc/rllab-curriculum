import collections
import random

import numpy as np

from rllab.misc import logger


class Region(object):

    def __init__(self, min_border, max_border, max_history=100, max_goals=500):
        self.states = collections.deque(maxlen=max_history)
        self.competences = collections.deque(maxlen=max_history)
        self.min_border = min_border
        self.max_border = max_border
        self.num_goals = 0
        self.max_goals = max_goals
        self.max_history = max_history

    # Add this state and competence to the region.
    def add_state(self, state, competence):
        self.states.append(state)
        self.competences.append(competence)
        self.num_goals += 1

    def is_too_big(self):
        return self.num_goals > self.max_goals

    # Split this region into subregions.
    def split(self):
        #TODO - perform a smart split.
        # For now, just perform a single split.
        region1_min = np.copy(self.min_border)
        region1_max = np.copy(self.max_border)
        region1_max[0] = (self.min_border[0] + self.max_border[0])/2 # Cut the first dimension in half.
        region1 = Region(region1_min, region1_max, max_history=self.max_history, max_goals=self.max_goals)

        region2_min = np.copy(self.min_border)
        region2_min[0] = (self.min_border[0] + self.max_border[0])/2 # Cut the first dimension in half.
        region2_max = np.copy(self.max_border)
        region2 = Region(region2_min, region2_max, max_history=self.max_history, max_goals=self.max_goals)

        # Reassign all goals to one of these regions.
        for state, competence in zip(self.states, self.competences):
            if region1.contains(state):
                region1.add_state(state, competence)
            elif region2.contains(state):
                region2.add_state(state, competence)
            else:
                logger.log("Region 1: " + str(region1.min_border) + " " + str(region1.max_border))
                logger.log("Region 2: " + str(region2.min_border) + " " + str(region2.max_border))
                raise Exception("Split region; now cannot find region for state: " + str(state))

        return [region1, region2]


    # Compute the sum of the competences in a given range.
    def compute_local_measure(self, start_index, end_index):
        return np.sum(np.array(self.competences)[start_index:end_index])

    # Compute the derivative of competences.
    def compute_interest(self):
        num_states = len(self.states)
        old_measure = self.compute_local_measure(0, int(num_states/2) - 1)
        new_measure = self.compute_local_measure(int(num_states/2) + 1, num_states-1)
        interest = abs(old_measure - new_measure) / num_states
        return interest

    # Check whether this state is inside this region.
    def contains(self, state):
        return (self.min_border.tolist() < list(state) < self.max_border.tolist())

    def sample_uniform(self):
        state = []
        for min_val, max_val in zip(self.min_border, self.max_border):
            state.append(random.uniform(min_val, max_val))
        return state


class SaggRIAC(object):

    def __init__(self, state_size, state_range=None, state_center=None, state_bounds=None, max_history=100, max_goals=500):
        self.max_goals = max_goals

        self.regions = []

        self.state_size = state_size
        self.state_center = np.array(state_center) if state_center is not None else np.zeros(state_size)
        if state_range is not None:
            self.state_range = state_range
            self.state_bounds = np.vstack([self.state_center - self.state_range * np.ones(self.state_size),
                                           self.state_center + self.state_range * np.ones(self.state_size)])
        elif state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
            self.state_range = self.state_bounds[1] - self.state_bounds[0]

        min_border = self.state_bounds[0]
        max_border = self.state_bounds[1]

        # Create a region to represent the entire space.
        whole_region = Region(min_border, max_border, max_history=max_history, max_goals=self.max_goals)
        self.regions.append(whole_region)

    # Find the region that contains a given state.
    def find_region(self, state):
        for index, region in enumerate(self.regions):
            if region.contains(state):
                return [index, region]
        raise Exception("Cannot find state: " + str(state) + " in any region!")

    # Add these states and competences to our list.
    def add_states(self, states, competences):
        for state, competence in zip(states, competences):
            # Find the appropriate region for this state.
            index, region = self.find_region(state)
            # Add this state to the region.
            region.add_state(state, competence)

            # If the region contains too many goals, split it into subregions.
            if region.is_too_big():
                [region1, region2] = region.split()
                # Add the subregions and delete the original region.
                self.regions.append(region1)
                self.regions.append(region2)
                del self.regions[index]

    # Sample states from the regions.
    def sample_states(self, num_samples):
        # Mode 1
        samples = self.sample_mode_1(num_samples)

        # TODO - Modes 2 and 3
        return samples

    # Temporary hack - just randomly pick a region to sample from.
    def sample_random_region(self, num_samples):
        samples = []
        for i in range(num_samples):
            region_index = random.randrange(len(self.regions))
            region = self.regions[region_index]
            state = region.sample_uniform()
            samples.append(state)
        return samples

    def sample_mode_1(self, num_samples):
        if len(self.regions) == 1:
            return self.sample_random_region(num_samples)

        interests = []
        for region in self.regions:
            interests.append(region.compute_interest())

        # Subtract the min interest
        min_interest = min(interests)
        interests -= min_interest

        sum_interests = sum(interests)
        if sum_interests == 0:
            logger.log("All interests are " + str(min_interest))
            return self.sample_random_region(num_samples)

        # Normalize
        interests /= sum_interests

        probs = interests
        num_per_regions = np.random.multinomial(num_samples, probs)

        samples = []
        for region_index, num_per_region in enumerate(num_per_regions):
            region = self.regions[region_index]
            for i in range(num_per_region):
                sample = region.sample_uniform()
                samples.append(sample)

        return samples
