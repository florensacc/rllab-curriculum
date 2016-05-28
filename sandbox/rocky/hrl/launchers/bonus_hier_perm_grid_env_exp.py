from __future__ import print_function
from __future__ import absolute_import

import sys
import os
# os.environ["THEANO_FLAGS"] = "device=cpu"
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
# from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.bonus_evaluators.marginal_parsimony_bonus_evaluator import MarginalParsimonyBonusEvaluator
from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES
from sandbox.rocky.hrl.density_estimators.gmm_density_estimator import GMMDensityEstimator
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from sandbox.rocky.hrl.algos.bonus_algos import BonusTRPO


stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("grid_size", [5])#, 9])
vg.add("batch_size", [4000])#1000])#20000])
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("bonus_coeff", [0.0])#0.1])#0., 0.001, 0.01, 0.1, 1.0, 10.0])
vg.add("bottleneck_coeff", [0.0])#0.1])#0., 0.001, 0.01, 0.1, 1.0, 10.0])
vg.add("use_trust_region", [False])
vg.add("step_size", [0.])
vg.add("use_decision_nodes", [False])
vg.add("random_reset", [False])
vg.add("use_bottleneck", [True])
vg.add("bottleneck_dim", [7, 5, 3, 1])#
vg.add("mode", [
    # MODES.MODE_MARGINAL_PARSIMONY,
    # MODES.MODE_JOINT_MI_PARSIMONY,
    MODES.MODE_MI_FEUDAL_SYNC,
    # MODES.MODE_MI_FEUDAL,
    # MODES.MODE_HIDDEN_AWARE_PARSIMONY,
])

variants = vg.variants()
print("#Experiments:", len(variants))

for v in variants:
    env = PermGridEnv(size=v["grid_size"], n_objects=v["grid_size"], object_seed=0, random_restart=False)
    policy = StochasticGRUPolicy(
        env_spec=env.spec,
        n_subgoals=v["grid_size"],
        use_decision_nodes=v["use_decision_nodes"],
        random_reset=v["random_reset"],
        use_bottleneck=v["use_bottleneck"],
        bottleneck_dim=v["bottleneck_dim"],
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    # bonus_evaluator = MarginalParsimonyBonusEvaluator(
    #     env_spec=env.spec,
    #     policy=policy,
    #     bonus_coeff=v["bonus_coeff"],
    #     regressor_args=dict(
    #         use_trust_region=v["use_trust_region"],
    #         step_size=v["step_size"],
    #     )
    # )
    bonus_evaluator = DiscreteBonusEvaluator(
        env_spec=env.spec,
        policy=policy,
        mode=v["mode"],
        bonus_coeff=v["bonus_coeff"],
        bottleneck_coeff=v["bottleneck_coeff"],
        regressor_args=dict(
            use_trust_region=v["use_trust_region"],
            step_size=v["step_size"],
        ),
        bottleneck_density_estimator_cls=GMMDensityEstimator,
        bottleneck_density_estimator_args=dict(),
    )
    algo = BonusTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v["batch_size"],
        bonus_evaluator=bonus_evaluator,
        step_size=0.01,
        max_path_length=100,
        n_itr=100,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="hrl_bottleneck",
        n_parallel=1,
        seed=v["seed"],
        mode="local",
    )
    sys.exit(0)
