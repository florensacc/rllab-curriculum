from __future__ import print_function
from __future__ import absolute_import

import sys
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.bonus_evaluators.marginal_parsimony_bonus_evaluator import MarginalParsimonyBonusEvaluator
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from sandbox.rocky.hrl.algos.bonus_algos import BonusTRPO


stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("grid_size", [5])#, 7, 9, 11])
vg.add("batch_size", [4000, 10000, 20000])#4000])#, 10000, 20000])
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("bonus_coeff", [0.1, 0.01, 0, 0.001, 1.0])
vg.add("use_trust_region", [False])#True, False])
vg.add("step_size", [0.])#lambda use_trust_region: [0.] if not use_trust_region else [0.01, 0.1, 1.0, 10.0])
vg.add("use_decision_nodes", [True, False])

variants = vg.variants()
print("#Experiments:", len(variants))

for v in variants:
    env = PermGridEnv(size=v["grid_size"], n_objects=v["grid_size"], object_seed=0)
    policy = StochasticGRUPolicy(
        env_spec=env.spec,
        n_subgoals=v["grid_size"],
        use_decision_nodes=False
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_evaluator = MarginalParsimonyBonusEvaluator(
        env_spec=env.spec,
        policy=policy,
        bonus_coeff=v["bonus_coeff"],
        regressor_args=dict(
            use_trust_region=v["use_trust_region"],
            step_size=v["step_size"],
        )
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
        exp_prefix="hier_marginal_parsimony_2",
        n_parallel=1,
        seed=v["seed"],
        mode="lab_kube"
    )
    # sys.exit(0)

