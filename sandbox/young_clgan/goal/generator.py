import numpy as np

from sandbox.young_clgan.gan.gan import FCGAN
from sandbox.young_clgan.goal.utils import sample_matrix_row



class StateGenerator(object):
    """A base class for state generation."""
    
    def pretrain_uniform(self):
        """Pretrain the generator distribution to uniform distribution in the limit."""
        raise NotImplementedError
        
    def pretain(self, goals):
        """Pretrain with goal distribution in the goals list."""
        raise NotImplementedError
    
    def sample_states(self, size):
        """Sample goals with given size."""
        raise NotImplementedError
    
    def sample_states_with_noise(self, size):
        """Sample goals with noise."""
        raise NotImplementedError
        
    def train(self, goals, labels):
        """Train with respect to given goals and labels."""
        raise NotImplementedError
        

class CrossEntropyStateGenerator(StateGenerator):
    """Maintain a state list and add noise to current goals to generate new goals."""
    
    def __init__(self, goal_size, evaluater_size, goal_range, noise_std=1.0,
                 goal_center=None):
        self.goal_list = np.array([])
        self.goal_range = goal_range
        self.noise_std = noise_std
        self.goal_center = np.array(goal_center) if goal_center is not None else np.zeros(goal_size)
        
    def pretrain_uniform(self, size=1000):
        goals = self.goal_center + np.random.uniform(
            -self.goal_range, self.goal_range, size=(size, self.goal_size)
        )
        return self.pretrain(goals)
        
    def pretain(self, goals):
        self.goal_list = np.array(goals)
        
    def sample_states(self, size):
        if len(self.goal_list) == 0:
            raise ValueError('Generator uninitialized!')
        
        goals = sample_matrix_row(self.goal_list, size)
        return np.clip(
            goals + np.random.randn(*goals.shape) * self.noise_std,
            -self.goal_range, self.goal_range
        )
    
    def sample_states_with_noise(self, size):
        return self.sample_states(size)
        
    def train(self, goals, labels):
        labels = np.mean(labels, axis=1) >= 1
        good_goals = np.array(goals)[labels, :]
        if len(good_goals) != 0:
            self.goal_list = good_goals
        


class StateGAN(StateGenerator):
    """A GAN for generating states. It is just a wrapper for clgan.GAN.FCGAN"""

    def __init__(self, goal_size, evaluater_size, goal_range,
                 goal_noise_level, goal_center=None, *args, **kwargs):
        self.gan = FCGAN(
            generator_output_size=goal_size,
            discriminator_output_size=evaluater_size,
            *args,
            **kwargs
        )
        self.goal_size = goal_size
        self.evaluater_size = evaluater_size
        self.goal_range = goal_range
        self.goal_center = np.array(goal_center) if goal_center is not None else np.zeros(goal_size)
        self.goal_noise_level = goal_noise_level
        print('goal_center is : ', self.goal_center, 'goal_range: ', self.goal_range)

    def pretrain_uniform(self, size=10000, *args, **kwargs):
        """
        :param size: number of uniformly sampled states (that we will try to fit as output of the GAN)
        :param outer_iters: of the GAN
        """
        goals = self.goal_center + np.random.uniform(
            -self.goal_range, self.goal_range, size=(size, self.goal_size)
        )
        return self.pretrain(goals, *args, **kwargs)

    def pretrain(self, goals, *args, **kwargs):
        """
        Pretrain the goal GAN to match the distribution of given goals.
        :param goals: the goal distribution to match
        :param outer_iters: of the GAN
        """
        labels = np.ones((goals.shape[0], self.evaluater_size))  # all goal same label --> uniform
        return self.train(goals, labels, *args, **kwargs)

    def _add_noise_to_goals(self, goals):
        noise = np.random.randn(*goals.shape) * self.goal_noise_level
        goals += noise
        return np.clip(goals, self.goal_center - self.goal_range, self.goal_center + self.goal_range)

    def sample_states(self, size):  # un-normalizes the goals
        normalized_goals, noise = self.gan.sample_generator(size)
        goals = self.goal_center + normalized_goals * self.goal_range
        return goals, noise

    def sample_states_with_noise(self, size):
        goals, noise = self.sample_states(size)
        goals = self._add_noise_to_goals(goals)
        return goals, noise

    def train(self, goals, labels, suppress_generated_goals=True, *args, **kwargs):
        normalized_goals = (goals - self.goal_center) / self.goal_range
        return self.gan.train(
            normalized_goals, labels, suppress_generated_states=suppress_generated_goals, *args, **kwargs)

    def discriminator_predict(self, goals):
        return self.gan.discriminator_predict(goals)
