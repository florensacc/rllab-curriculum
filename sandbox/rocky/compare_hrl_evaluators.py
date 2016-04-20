from __future__ import absolute_import
from __future__ import print_function

import joblib

from rllab.sampler import parallel_sampler

parallel_sampler.config_parallel_sampler(n_parallel=4, base_seed=0)
from sandbox.rocky.hrl.mi_evaluator.exact_state_based_mi_evaluator import ExactStateBasedMIEvaluator
from sandbox.rocky.hrl.mi_evaluator.state_based_mi_evaluator import StateBasedMIEvaluator
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab import hrl_utils
from rllab.misc import logger

data = joblib.load("data/local/hrl-level1/hrl_level1_2016_04_08_15_45_02_0001/params.pkl")
env = data["env"]
policy = data["policy"]

exact_evaluator = ExactStateBasedMIEvaluator(
    env=env,
    policy=policy,
)

approx_evaluator = StateBasedMIEvaluator(
    env_spec=env.spec,
    policy=policy,
    regressor_cls=CategoricalMLPRegressor,
    regressor_args=dict(
        use_trust_region=False,
        hidden_sizes=tuple(),
        optimizer=LbfgsOptimizer(max_opt_itr=200),
    )
)

N_SAMPLES = 36000
MAX_PATH_LENGTH = 60

parallel_sampler.populate_task(env, policy)
parallel_sampler.request_samples(policy.get_param_values(), max_samples=N_SAMPLES, max_path_length=MAX_PATH_LENGTH)

paths = parallel_sampler.collect_paths()

high_paths = []
full_high_paths = []
low_paths = []
bonus_returns = []
# Collect high-level trajectories
for path in paths:
    rewards = path['rewards']
    high_agent_infos = path["agent_infos"]["high"]
    high_observations = path["agent_infos"]["high_obs"]
    subgoals = path["agent_infos"]["subgoal"]
    high_path = dict(
        observations=high_observations,
        actions=subgoals,
        env_infos=path["env_infos"],
        agent_infos=high_agent_infos,
        rewards=rewards,
    )
    chunked_high_path = hrl_utils.subsample_path(high_path, policy.subgoal_interval)
    high_paths.append(chunked_high_path)
    full_high_paths.append(high_path)

# We need to train the predictor for p(s'|g, s)
exact_evaluator.fit(high_paths)
exact_evaluator.update_cache()

logger.log("fitting approximate evaluator")
approx_evaluator.fit(high_paths)

print("Exact MI:", exact_evaluator.mi_avg)
print("Predicted MI:", approx_evaluator.get_predicted_mi(env, policy))#, approx_evaluatorexact_evaluator._mi_avg)
