import numpy as np
from rllab.algos.trpo import TRPO
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.young_clgan.goal.evaluator import evaluate_goal_env
from sandbox.young_clgan.lib.envs.base import UniformGoalGenerator


class TRPOGoal(TRPO):
    """
    Trust Region Policy Optimization
    """

    @overrides
    def log_diagnostics(self, paths):
        super(TRPOGoal, self).log_diagnostics(paths)
        self.env.reset(fix_goal=self.env.final_goal)
        old_goal_generator = self.env.goal_generator
        uniform_goal_generator = UniformGoalGenerator(goal_size=np.size(self.env.final_goal),
                                                      bounds=self.env.feasible_goal_space.bounds)
        # fixed_goal_generator = FixedGoalGenerator(goal=self.env.final_goal)
        logger.log("Evaluating performance on Unif and Fix Goal Gen...")
        # log some more on how the pendulum performs the upright and general task
        with logger.tabular_prefix('UnifGoalGen_'):
            self.env.update_goal_generator(goal_generator=uniform_goal_generator)
            evaluate_goal_env(self.env, policy=self.policy, horizon=self.max_path_length, n_goals=10)
        # with logger.tabular_prefix('FixGoalGen_'):
        #     self.env.update_goal_generator(goal_generator=fixed_goal_generator)
        #     evaluate_goal_env(self.env, policy=self.policy, horizon=self.max_path_length, n_goals=10)
        self.env.update_goal_generator(goal_generator=old_goal_generator)



