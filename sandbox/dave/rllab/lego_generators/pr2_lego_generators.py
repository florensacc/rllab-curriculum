from __future__ import absolute_import
from __future__ import print_function
import time
from cgitb import small
from datetime import datetime

from sandbox.dave.rllab.goal_generators.goal_generator import BoxGoalGenerator, GoalGenerator, FixedGoalGenerator
import numpy as np
from six.moves import range


class PR2LegoSmallRange:
    def __init__(self):
        # Initial tip position: [0.94, 0.17, 0.79]
        self.low_lego_bounds = np.array([0.55, 0.3, 0.5025, 1, 0, 0, 0])    #(0.45  0.1
        self.high_lego_bounds = np.array([0.7, 0.45, 0.5025, 1, 0, 0, 0])   #(0.65  0.4


class PR2LegoLargeRange:
    def __init__(self):
        # Initial tip position: [0.94, 0.17, 0.79]
        # low_goal_bounds = np.array([0.3, -0.3, 0.3]) # Bad - contained unreachable goals
        # high_goal_bounds = np.array([0.8, 0.7, 1.3]) # Bad - contained unreachable goals
        self.low_lego_bounds = np.array([0.45, 0.25, 0.5025, 1, 0, 0, 0])
        self.high_lego_bounds = np.array([0.75, -0.25, 0.5025, 1, 0, 0, 0])


class PR2LegoFixedBlockGenerator(FixedGoalGenerator):
    """ Always returns the same fixed goal """
    def __init__(self, block=None):
        # print "Using fixed goal generator"
        # import pdb
        # pdb.set_trace()
        if block is None:
            # block = (0.6, 0.2, 0.5025, 1, 0, 0, 0)
            block = (0.6, 0.35, 0.5025, 1, 0, 0, 0)
        block = np.array(block)
        super(PR2LegoFixedBlockGenerator, self).__init__(block)


class PR2LegoBoxBlockCurriculum(BoxGoalGenerator):
    def __init__(
            self,
            distance_thresh=0.01,  # 1 cm
            init_size=0.01,
            update_delta=0.1,
            target_paths_within_thresh=0.9,
            nonlinearity=10,
            small_range=True
    ):
        self.distance_thresh = distance_thresh
        self.update_delta = update_delta
        self.init_size = init_size
        self.target_paths_within_thresh = target_paths_within_thresh
        self.nonlinearity = nonlinearity

        if small_range:
            lego_range = PR2LegoSmallRange()
        else:
            lego_range = PR2LegoLargeRange()

        # Largest allowed values for the box.
        self.min_low_lego_bounds = lego_range.low_lego_bounds
        self.max_high_lego_bounds = lego_range.high_lego_bounds
        assert(np.less(self.min_low_lego_bounds, self.max_high_lego_bounds).all())

        # Initial range for the box.
        mean_lego = np.mean([self.min_low_lego_bounds, self.max_high_lego_bounds], axis=0)
        self.init_low_lego_bounds = mean_lego - self.init_size / 2
        self.init_high_lego_bounds = mean_lego + self.init_size / 2
        assert(np.less_equal(self.init_low_lego_bounds, self.init_high_lego_bounds).all())

        # Set the current range to the initial range.
        self.low_lego_bounds = self.init_low_lego_bounds
        self.high_lego_bounds = self.init_high_lego_bounds
        assert(np.less_equal(self.low_lego_bounds, self.high_lego_bounds).all())

        # Ensure that the box never gets smaller than the initial range.
        self.max_low_lego_bounds = self.low_lego_bounds
        self.min_high_lego_bounds = self.high_lego_bounds
        assert(np.less_equal(self.max_low_lego_bounds, self.min_high_lego_bounds).all())
        assert(np.less_equal(self.min_low_lego_bounds, self.max_low_lego_bounds).all())
        assert(np.less_equal(self.min_high_lego_bounds, self.max_high_lego_bounds).all())

        # Ensure that the box stays within the acceptable range.
        self._limit_bounds()
        # Update the superclass.
        self._update_box()

    def _limit_bounds(self):
        # Limit the bounds to the largest allowed size.
        self.low_lego_bounds = np.maximum(self.low_lego_bounds, self.min_low_lego_bounds)
        self.high_lego_bounds = np.minimum(self.high_lego_bounds, self.max_high_lego_bounds)

        # Limit the bounds to the smallest allowed size.
        self.low_lego_bounds = np.minimum(self.low_lego_bounds, self.max_low_lego_bounds)
        self.high_lego_bounds = np.maximum(self.high_lego_bounds, self.min_high_lego_bounds)

        assert (np.less_equal(self.low_lego_bounds, self.high_lego_bounds).all())

    def _update_box(self):
        # Update the box.
        super(PR2LegoBoxBlockCurriculum, self).__init__(
            self.low_lego_bounds, self.high_lego_bounds)

    def update(self, paths):
        distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])

        score = (paths_within_thresh - self.target_paths_within_thresh) / (1 - self.target_paths_within_thresh)

        # score = (paths_within_thresh - self.target_paths_within_thresh)
        print(("Target paths within thresh: " + str(self.target_paths_within_thresh)))
        print(("Paths within thresh: " + str(paths_within_thresh)))
        print(("Score: " + str(score)))

        # Once we increase the world size, give the policy a chance before immediately reducing it.
        if score < 0:
            score /= self.nonlinearity

        # else:
        #     score /= (1 - self.target_paths_within_thresh)

        # Update the bounds.
        self.low_lego_bounds -= score * self.update_delta
        self.high_lego_bounds += score * self.update_delta

        # Ensure that the box stays within the acceptable range.
        self._limit_bounds()
        # Update the superclass.
        self._update_box()

        print(("Low bounds: " + str(self.low_lego_bounds)))
        print(("High bounds: " + str(self.high_lego_bounds)))

    def get_diagnostics(self):
        size_increase_low = np.mean(self.init_low_lego_bounds - self.low_lego_bounds)
        size_increase_high = np.mean(self.high_lego_bounds - self.init_high_lego_bounds)
        size_increase = np.mean([size_increase_low, size_increase_high])

        size_left_low = np.mean(self.low_lego_bounds - self.min_low_lego_bounds)
        size_left_high = np.mean(self.max_high_lego_bounds - self.high_lego_bounds)
        size_left = np.mean([size_left_low, size_left_high])

        diagnostics = dict()
        diagnostics["SizeIncrease"] = size_increase
        diagnostics["SizeLeft"] = size_left

        return diagnostics


class PR2LegoBoxBlockGenerator(BoxGoalGenerator):
    """ Generate goals randomly from within a box that defines the goal space """
    def __init__(self,
                 max_episodes_with_goal=1,
                 small_range=True):
        if small_range:
            self.goal_range = PR2LegoSmallRange()
        else:
            self.goal_range = PR2LegoLargeRange()

        low_lego_bounds = self.goal_range.low_lego_bounds
        high_lego_bounds = self.goal_range.high_lego_bounds

        super(PR2LegoBoxBlockGenerator, self).__init__(
            low_lego_bounds, high_lego_bounds,
            max_episodes_with_goal=max_episodes_with_goal)


class PR2LegoBoxBlockGeneratorSmall(BoxGoalGenerator):
    """ Generate goals randomly from within a box that defines the goal space """
    def __init__(self, max_episodes_with_goal=1):
        lego_range = PR2LegoSmallRange()

        low_lego_bounds = lego_range.low_lego_bounds
        high_lego_bounds = lego_range.high_lego_bounds

        super(PR2LegoBoxBlockGeneratorSmall, self).__init__(
            low_lego_bounds, high_lego_bounds,
            max_episodes_with_goal=max_episodes_with_goal)


class PR2LegoBoxBlockGeneratorLarge(BoxGoalGenerator):
    """ Generate goals randomly from within a box that defines the goal space """
    def __init__(self, max_episodes_with_goal=1):
        lego_range = PR2LegoLargeRange()

        low_lego_bounds = lego_range.low_lego_bounds
        high_lego_bounds = lego_range.high_lego_bounds

        super(PR2LegoBoxBlockGeneratorLarge, self).__init__(
            low_lego_bounds, high_lego_bounds,
            max_episodes_with_goal=max_episodes_with_goal)


class PR2LegoTestBlockGenerator(GoalGenerator):
    """ Generate a list of goals that the we test if the policy can achieve """
    def __init__(self, num_test_legos, obs=None, seed=0, small_range=False):
        self.legos = []
        # Set the seed so that we produce consistent results
        np.random.seed(seed)
        self.lego_generator = PR2LegoBoxBlockGenerator(small_range=small_range)
        self.num_test_legos = num_test_legos
        self.generate_all_legos(obs)
        self.lego_index = 0
        # Reset the random seed to get randomness
        # np.random.seed(time.time())
        # np.random.seed(datetime.now())

    def generate_all_legos(self, obs):
        for _ in range(self.num_test_legos):
            lego = self.lego_generator.generate_goal(obs)
            self.legos.append(lego)

    def generate_goal(self, obs):
        goal = self.legos[self.lego_index]
        self.lego_index += 1
        return goal
