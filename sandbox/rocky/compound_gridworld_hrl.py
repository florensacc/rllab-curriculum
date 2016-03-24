import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.algos.batch_hrl import BatchHRL
from rllab.spaces import Discrete
from rllab.envs.subgoal_env import SubgoalEnv
from rllab.policies.subgoal_policy import SubgoalPolicy
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.baselines.subgoal_baseline import SubgoalBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.mi_evaluator.state_given_goal_mi_evaluator import StateGivenGoalMIEvaluator
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.misc import ext

# stub(globals())


env = GridWorldEnv(
    desc=[
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFF"
    ]
)

env = CompoundActionSequenceEnv(
    wrapped_env=env,
    reset_history=True,
    # obs_include_actions=True,
    action_map=[
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],

        # [0, 1, 2],
        # [0, 0, 0],
        # [2, 1, 0],
        # [3, 3, 3],


        # [0],
        # [1],
        # [2],
        # [3],
    ]
)

env = SubgoalEnv(
    wrapped_env=env,
    subgoal_space=Discrete(n=4),
    # low_action_history_length=0,
)

algo = BatchHRL(
    batch_size=10000,
    whole_paths=True,
    max_path_length=60,
    n_itr=100,
    high_algo=TRPO(
        discount=0.99,
        step_size=0.01,
    ),
    low_algo=TRPO(
        discount=0.99,
        step_size=0.01,
    ),
)

high_policy = CategoricalMLPPolicy(
    env_spec=env.spec.high_env_spec,
)

low_policy = CategoricalMLPPolicy(
    env_spec=env.spec.low_env_spec,
)

policy = SubgoalPolicy(
    env_spec=env.spec,
    subgoal_interval=3,
    high_policy=high_policy,
    low_policy=low_policy,
)

baseline = SubgoalBaseline(
    env_spec=env.spec,
    high_baseline=GaussianMLPBaseline(
        env_spec=env.spec.high_env_spec,
        regressor_args=dict(
            optimizer=ConjugateGradientOptimizer(),
        ),
    ),
    low_baseline=GaussianMLPBaseline(
        env_spec=env.spec.low_env_spec,
        regressor_args=dict(
            optimizer=ConjugateGradientOptimizer(),
        ),
    ),
)

evaluator = StateGivenGoalMIEvaluator(
    env_spec=env.spec,
    high_policy_dist_family=high_policy.distribution,
    low_policy_dist_family=low_policy.distribution,
    regressor_cls=CategoricalMLPRegressor,
    regressor_args=dict(
        optimizer=ConjugateGradientOptimizer(),
    ),
)

run_experiment_lite(
    algo.train(env=env, policy=policy, baseline=baseline, bonus_evaluator=evaluator),
    exp_prefix="compound_gridworld_4x4_free_hrl",
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    mode="local",
    # env=ext.merge_dict(os.environ, dict(THEANO_FLAGS="optimizer=None"))
)
