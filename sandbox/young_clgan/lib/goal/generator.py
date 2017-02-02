import numpy as np

from sandbox.young_clgan.lib.gan.gan import FCGAN


class GoalGAN(object):
    """A GAN for generating goals. It is just a wrapper for clgan.GAN.FCGAN"""

    def __init__(self, goal_size, evaluater_size, goal_range,
                 goal_noise_level, *args, **kwargs):
        self.gan = FCGAN(
            generator_output_size=goal_size,
            discriminator_output_size=evaluater_size,
            *args,
            **kwargs
        )
        self.goal_size = goal_size
        self.evaluater_size = evaluater_size
        self.goal_range = goal_range
        self.goal_noise_level = goal_noise_level

    def pretrain_uniform(self, size=10000, outer_iters=10, generator_iters=5,
                         discriminator_iters=200):
        """
        :param size: number of uniformly sampled goals (that we will try to fit as output of the GAN)
        :param outer_iters: of the GAN
        """
        goals = np.random.uniform(
            -self.goal_range, self.goal_range, size=(size, self.goal_size)
        )
        labels = np.ones((size, self.evaluater_size))  # all goal same label --> uniform
        self.train(
            goals, labels, outer_iters, generator_iters, discriminator_iters, suppress_generated_goals=True
        )

    def _add_noise_to_goals(self, goals):
        noise = np.random.randn(*goals.shape) * self.goal_noise_level
        goals += noise
        return np.clip(goals, -self.goal_range, self.goal_range)

    def sample_goals(self, size):
        goals, noise = self.gan.sample_generator(size)
        goals = goals * self.goal_range
        return goals, noise

    def sample_goals_with_noise(self, size):
        goals, noise = self.sample_goals(size)
        goals = self._add_noise_to_goals(goals)
        return goals, noise

    def train(self, goals, labels, outer_iters, generator_iters,
              discriminator_iters, suppress_generated_goals=False):
        goals = goals / self.goal_range
        self.gan.train(
            goals, labels, outer_iters, generator_iters, discriminator_iters, suppress_generated_goals
        )

    def discriminator_predict(self, goals):
        return self.gan.discriminator_predict(goals)
