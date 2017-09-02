import numpy as np


""" Reward function utilities."""

def linear_threshold_reward(distance, threshold, coefficient):
    """
    Linear reward with respect to distance, cut of at threshold.
    :param distance: current distance to the goal
    :param threshold: maximum distance at which some bonus is given
    :param coefficient: NEGATIVE --> slope of the linear bonus
    """
    assert distance >= 0 and threshold > 0 and coefficient < 0
    constant = -threshold * coefficient
    return max(distance * coefficient + constant, 0)


def gaussian_threshold_reward(distance, threshold, alpha, beta):
    """Gaussian reward with respect to distance, cut of at threshold."""
    assert distance >= 0 and threshold > 0 and alpha > 0 and beta > 0
    if distance > threshold:
        return 0

    return alpha * np.exp(-distance / beta)
