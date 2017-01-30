from __future__ import absolute_import
from __future__ import print_function
from sandbox.dave.rllab.goal_generators.goal_generator \
    import BoxGoalGenerator, GoalGenerator, FixedGoalGenerator, CrownGoalGenerator
import numpy as np
from six.moves import range


class PR2CrownSmallRange:
    def __init__(self):
        # Initial tip position: [0.94, 0.17, 0.79]
        self.radius_low = 0.1   #TODO: this is a fixGoal generator
        self.radius_high = 0.2

class PR2CrownLargeRange:
    def __init__(self):
        # Initial tip position: [0.94, 0.17, 0.79]
        self.radius_low = 0.05  #TODO: this is a fixGoal generator
        self.radius_high = 0.35


class PR2SmallRange:
    def __init__(self):
        # Initial tip position: [0.94, 0.17, 0.79]
        self.low_goal_bounds = np.array([0.4, 0.5, 0.5025])   #(0.4, -0.1, )       #TODO: this is a fixGoal generator
        self.high_goal_bounds = np.array([0.6, 0.2, 0.5025])  #(0.7, 0.5, )


class PR2LargeRange:
    def __init__(self):
        # Initial tip position: [0.94, 0.17, 0.79]
        # low_goal_boundhuntest12
        # s = np.array([0.3, -0.3, 0.3]) # Bad - contained unreachable goals
        # high_goal_bounds = np.array([0.8, 0.7, 1.3]) # Bad - contained unreachable goals
        self.low_goal_bounds = np.array([0.4, -0.2, 0.5025])
        self.high_goal_bounds = np.array([0.8, 0.8, 1])


class PR2FixedGoalGenerator(FixedGoalGenerator):
    """ Always returns the same fixed goal """
    def __init__(self, goal=None):
        # print "Using fixed goal generator"
        if goal is None:
            # goal = (0.5, 0.3, 0.5025)
            goal = (0.6, 0., 0.5025)
            #goal = (0.8, 0.17, 1.0)
        goal = np.array(goal)
        super(PR2FixedGoalGenerator, self).__init__(goal)


class PR2BoxGoalCurriculum(BoxGoalGenerator):
    def __init__(
            self,
            distance_thresh=0.01, # 1 cm
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
            goal_range = PR2SmallRange()
        else:
            goal_range = PR2LargeRange()

        # Largest allowed values for the box.
        self.min_low_goal_bounds = goal_range.low_goal_bounds
        self.max_high_goal_bounds = goal_range.high_goal_bounds
        assert(np.less(self.min_low_goal_bounds, self.max_high_goal_bounds).all())

        # Initial range for the box.
        mean_goal = np.mean([self.min_low_goal_bounds, self.max_high_goal_bounds], axis=0)
        self.init_low_goal_bounds = mean_goal - self.init_size / 2
        self.init_high_goal_bounds = mean_goal + self.init_size / 2
        assert(np.less_equal(self.init_low_goal_bounds, self.init_high_goal_bounds).all())

        # Set the current range to the initial range.
        self.low_goal_bounds = self.init_low_goal_bounds
        self.high_goal_bounds = self.init_high_goal_bounds
        assert(np.less_equal(self.low_goal_bounds, self.high_goal_bounds).all())

        # Ensure that the box never gets smaller than the initial range.
        self.max_low_goal_bounds = self.low_goal_bounds
        self.min_high_goal_bounds = self.high_goal_bounds
        assert(np.less_equal(self.max_low_goal_bounds, self.min_high_goal_bounds).all())
        assert(np.less_equal(self.min_low_goal_bounds, self.max_low_goal_bounds).all())
        assert(np.less_equal(self.min_high_goal_bounds, self.max_high_goal_bounds).all())

        # Ensure that the box stays within the acceptable range.
        self._limit_bounds()
        # Update the superclass.
        self._update_box()

    def _limit_bounds(self):
        # Limit the bounds to the largest allowed size.
        self.low_goal_bounds = np.maximum(self.low_goal_bounds, self.min_low_goal_bounds)
        self.high_goal_bounds = np.minimum(self.high_goal_bounds, self.max_high_goal_bounds)

        # Limit the bounds to the smallest allowed size.
        self.low_goal_bounds = np.minimum(self.low_goal_bounds, self.max_low_goal_bounds)
        self.high_goal_bounds = np.maximum(self.high_goal_bounds, self.min_high_goal_bounds)

        assert (np.less_equal(self.low_goal_bounds, self.high_goal_bounds).all())

    def _update_box(self):
        # Update the box.
        super(PR2BoxGoalCurriculum, self).__init__(
            self.low_goal_bounds, self.high_goal_bounds)

    def update(self, paths):
        distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])

        score = (paths_within_thresh - self.target_paths_within_thresh) / (1 - self.target_paths_within_thresh)

        # score = (paths_within_thresh - self.target_paths_within_thresh)
        print("Target paths within thresh: " + str(self.target_paths_within_thresh))
        print("Paths within thresh: " + str(paths_within_thresh))
        print("Score: " + str(score))

        # Once we increase the world size, give the policy a chance before immediately reducing it.
        if score < 0:
            score /= self.nonlinearity

        # else:
        #     score /= (1 - self.target_paths_within_thresh)

        # Update the bounds.
        self.low_goal_bounds -= score * self.update_delta
        self.high_goal_bounds += score * self.update_delta

        # Ensure that the box stays within the acceptable range.
        self._limit_bounds()
        # Update the superclass.
        self._update_box()

        print("Low bounds: " + str(self.low_goal_bounds))
        print("High bounds: " + str(self.high_goal_bounds))

    def get_diagnostics(self):
        size_increase_low = np.mean(self.init_low_goal_bounds - self.low_goal_bounds)
        size_increase_high = np.mean(self.high_goal_bounds - self.init_high_goal_bounds)
        size_increase = np.mean([size_increase_low, size_increase_high])

        size_left_low = np.mean(self.low_goal_bounds - self.min_low_goal_bounds)
        size_left_high = np.mean(self.max_high_goal_bounds - self.high_goal_bounds)
        size_left = np.mean([size_left_low, size_left_high])

        diagnostics = dict()
        diagnostics["SizeIncrease"] = size_increase
        diagnostics["SizeLeft"] = size_left

        return diagnostics


class PR2BoxGoalGenerator(BoxGoalGenerator):
    """ Generate goals randomly from within a box that defines the goal space """
    def __init__(self,
                 max_episodes_with_goal=1,
                 small_range=True):
        if small_range:
            self.goal_range = PR2SmallRange()
        else:
            self.goal_range = PR2LargeRange()

        low_goal_bounds = self.goal_range.low_goal_bounds
        high_goal_bounds = self.goal_range.high_goal_bounds

        super(PR2BoxGoalGenerator, self).__init__(
            low_goal_bounds, high_goal_bounds,
            max_episodes_with_goal=max_episodes_with_goal)


class PR2BoxGoalGeneratorSmall(BoxGoalGenerator):
    """ Generate goals randomly from within a box that defines the goal space """
    def __init__(self, max_episodes_with_goal=1):
        goal_range = PR2SmallRange()

        low_goal_bounds = goal_range.low_goal_bounds
        high_goal_bounds = goal_range.high_goal_bounds

        super(PR2BoxGoalGeneratorSmall, self).__init__(
            low_goal_bounds, high_goal_bounds,
            max_episodes_with_goal=max_episodes_with_goal)


class PR2BoxGoalGeneratorLarge(BoxGoalGenerator):
    """ Generate goals randomly from within a box that defines the goal space """
    def __init__(self, max_episodes_with_goal=1):
        goal_range = PR2LargeRange()

        low_goal_bounds = goal_range.low_goal_bounds
        high_goal_bounds = goal_range.high_goal_bounds

        super(PR2BoxGoalGeneratorLarge, self).__init__(
            low_goal_bounds, high_goal_bounds,
            max_episodes_with_goal=max_episodes_with_goal)


class PR2TestGoalGenerator(GoalGenerator):
    """ Generate a list of goals that the we test if the policy can achieve """
    def __init__(self, range=0.4, delta=0.01, obs=None, seed=0, small_range=False):
        self.goals = []
        # Set the seed so that we produce consistent results
        np.random.seed(seed)
        self.goal_generator = PR2BoxGoalGenerator(small_range=small_range)
        self.range = range
        self.delta = delta
        self.goal_index = -1
        self.generate_all_goals()
        # Reset the random seed to get randomness
        # np.random.seed(time.time())
        # np.random.seed(datetime.now())

    def generate_all_goals(self):
        num_goals_x = np.int(np.round(self.range/self.delta))
        for i in range(num_goals_x):
            for j in range(num_goals_x):
                goal = np.array([j * self.delta - self.range/2, i * self.delta - self.range/2, 0])
                self.goals.append(goal)

    def generate_goal(self, obs):
        if self.goal_index != -1:
            goal = self.goals[self.goal_index]
        else:
            goal = self.goals[0]
        goal = goal + obs
        self.goal_index += 1
        return goal


class PR2CrownGoalGeneratorSmall(CrownGoalGenerator):
    """ Generate goals randomly from within a crown that defines the goal space """
    def __init__(self, max_episodes_with_goal=1):
        goal_crown = PR2CrownSmallRange()
        radius_low = goal_crown.radius_low
        radius_high = goal_crown.radius_high

        super(PR2CrownGoalGeneratorSmall, self).__init__(
            radius_low, radius_high,
            max_episodes_with_goal=max_episodes_with_goal)

class PR2CrownGoalGeneratorLarge(CrownGoalGenerator):
    """ Generate goals randomly from within a crown that defines the goal space """
    def __init__(self, max_episodes_with_goal=1):
        goal_crown = PR2CrownLargeRange()
        radius_low = goal_crown.radius_low
        radius_high = goal_crown.radius_high

        super(PR2CrownGoalGeneratorLarge, self).__init__(
            radius_low, radius_high,
            max_episodes_with_goal=max_episodes_with_goal)
