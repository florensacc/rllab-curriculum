from __future__ import print_function
import numpy as np

class BaseActionLimiter(object):
    def __init__(self,
                 distance_thresh=0.01,  # 1 cm
                 mean_failure_rate_init=1.0,
                 use_running_average_failure_rate=False,
                 failure_rate_gamma=0.9,
                 ):
        self.failure_rate_gamma = failure_rate_gamma
        self.distance_thresh = distance_thresh
        self.mean_failure_rate = mean_failure_rate_init
        self.use_running_average_failure_rate = use_running_average_failure_rate

    def get_action_limit(self):
        return self.action_limit

    def get_diagnostics(self):
        diagnostics = dict()
        diagnostics["ActionLimit"] = self.action_limit
        diagnostics["Mean Failure Rate"] = self.mean_failure_rate
        return diagnostics

    def update(self, paths, delta_pf):
        self.update_failure_rate(paths)

    def update_failure_rate(self, paths):
        distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])
        failure_rate = 1 - paths_within_thresh
        if self.use_running_average_failure_rate:
            self.mean_failure_rate = self.mean_failure_rate * self.failure_rate_gamma + failure_rate * (1 - self.failure_rate_gamma)
        else:
            # We don't want to be dependent on the initial failure rate, so just use a large batch size.
            self.mean_failure_rate = failure_rate

    def get_mean_failure_rate(self):
        return self.mean_failure_rate



class FixedActionLimiter(BaseActionLimiter):
    def __init__(
            self,
            distance_thresh=0.01,  # 1 cm
            action_limit=3,
            use_running_average_failure_rate=False,
            failure_rate_gamma=0.9,
    ):
        self.action_limit = action_limit
        super(FixedActionLimiter, self).__init__(
            use_running_average_failure_rate=use_running_average_failure_rate,
            failure_rate_gamma=failure_rate_gamma,
            distance_thresh=distance_thresh
        )

class UniformActionLimiter(BaseActionLimiter):
    def __init__(
            self,
            min_action_limit=0.1,
            max_action_limit=3,
    ):
        self.min_action_limit = min_action_limit
        self.max_action_limit = max_action_limit
        self.action_limit = min_action_limit

    def update(self, paths, delta_pdf):
        self.action_limit = np.random.uniform(self.min_action_limit, self.max_action_limit)


class AdaptiveActionLimiterBase(BaseActionLimiter):
    def __init__(self,
                 distance_thresh=0.01,  # 1 cm
                 min_action_limit=0.1,
                 max_action_limit=3,
                 use_running_average_failure_rate=False,
                 failure_rate_gamma=0.9,
                 ):
        self.min_action_limit = min_action_limit
        self.max_action_limit = max_action_limit
        self.action_limit = min_action_limit

        super(AdaptiveActionLimiterBase, self).__init__(use_running_average_failure_rate=use_running_average_failure_rate,
                                                        failure_rate_gamma=failure_rate_gamma,
                                                        distance_thresh=distance_thresh)

    def _limit_action_limit(self):
        # Make sure that the action limit stays within the specified range.
        self.action_limit = min(self.action_limit, self.max_action_limit)
        self.action_limit = max(self.action_limit, self.min_action_limit)

    def get_diagnostics(self):
        action_limit_increase = self.action_limit - self.min_action_limit
        action_limit_left = self.max_action_limit - self.action_limit

        diagnostics = dict()
        diagnostics["ActionLimitIncrease"] = action_limit_increase
        diagnostics["ActionLimitLeft"] = action_limit_left
        diagnostics["ActionLimit"] = self.action_limit
        diagnostics["Mean Failure Rate"] = self.mean_failure_rate

        return diagnostics



class AdaptiveActionLimiter(AdaptiveActionLimiterBase):
    def __init__(self,
                 distance_thresh=0.01,  # 1 cm
                 min_action_limit = 0.1,
                 max_action_limit = 3,
                 d_safe = 0.5,
                 action_limit_update_max=1.1,
                 use_running_average_failure_rate=False,
                 failure_rate_gamma=0.8
                 ):
        self.d_safe = d_safe
        self.action_limit_update_max = action_limit_update_max

        super(AdaptiveActionLimiter, self).__init__(
            distance_thresh=distance_thresh,
            min_action_limit=min_action_limit,
            max_action_limit=max_action_limit,
            use_running_average_failure_rate=use_running_average_failure_rate,
            failure_rate_gamma=failure_rate_gamma
        )

    def get_max_action_limit(self):
        return self.max_action_limit

    def update(self, paths, delta_pf):
        self.update_failure_rate(paths)

        #failure_rate = self.get_failure_rate(paths)
        # if failure_rate * self.action_limit > self.d_safe:
        #     # We are unsafe!  Reduce the action limit.
        #     self.action_limit = self.d_safe / failure_rate
        # else:

        predicted_failure_rate = self.get_mean_failure_rate() + delta_pf + 1e-5
        #self.action_limit = self.d_safe / predicted_failure_rate

        action_limit_update =  min(self.d_safe / (predicted_failure_rate * self.action_limit), self.action_limit_update_max)
        self.action_limit = action_limit_update * self.action_limit
        self._limit_action_limit()

        remaining_delta_pdf = self.d_safe / self.action_limit - predicted_failure_rate

        #failure_rate = self.get_failure_rate(paths) + delta_pf
        #safe_limit = self.d_safe / (failure_rate + 1e-5)

        #action_limit_update = self.d_safe / (failure_rate + 1e-5)

        #self.action_limit = safe_limit

        print("Action limit: " + str(self.action_limit))
        print("Remaining delta pdf: " + str(remaining_delta_pdf))
        return remaining_delta_pdf



class CurriculumActionLimiter(AdaptiveActionLimiterBase):
    def __init__(
            self,
            distance_thresh=0.01, # 1 cm
            min_action_limit=0.1,
            max_action_limit=3,
            update_delta=0.1,
            target_paths_within_thresh=0.9,
            nonlinearity=10,

    ):
        self.distance_thresh = distance_thresh
        self.update_delta = update_delta
        self.target_paths_within_thresh = target_paths_within_thresh
        self.nonlinearity = nonlinearity
        self.min_action_limit = min_action_limit
        self.max_action_limit = max_action_limit
        self.action_limit = min_action_limit

    def update(self, paths):
        distances_to_goal = [path["env_infos"]["distance_to_goal"] for path in paths]
        paths_within_thresh = np.mean([(d < self.distance_thresh).any() for d in distances_to_goal])

        if self.target_paths_within_thresh < 1:
            score = (paths_within_thresh - self.target_paths_within_thresh) / (1 - self.target_paths_within_thresh)
        else:
            score = (paths_within_thresh - self.target_paths_within_thresh)

        # score = (paths_within_thresh - self.target_paths_within_thresh)
        print("Target paths within thresh: " + str(self.target_paths_within_thresh))
        print("Paths within thresh: " + str(paths_within_thresh))
        print("Score: " + str(score))

        # Once we increase the world size, give the policy a chance before immediately reducing it.
        if score > 0:
            score /= self.nonlinearity

        # else:
        #     score /= (1 - self.target_paths_within_thresh)

        # Update the action limit.
        self.action_limit += score * self.update_delta

        # Ensure that the box stays within the acceptable range.
        self._limit_action_limit()

        print("Action limit: " + str(self.action_limit))

