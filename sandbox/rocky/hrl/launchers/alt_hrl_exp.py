


from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from sandbox.rocky.hrl.algos.alt_bonus_algos import AltBonusTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES

import sys

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("grid_size", [5, 7])  # , 7, 9, 11])
vg.add("batch_size", [])#20000])  # , 10000, 20000])
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("mode", [
    MODES.MODE_MI_FEUDAL_SYNC,
    # MODES.MODE_MI_FEUDAL,
    # MODES.MODE_BOTTLENECK_ONLY,
    # MODES.MODE_JOINT_MI_PARSIMONY,
    # MODES.MODE_MARGINAL_PARSIMONY,
    # MODES.MODE_HIDDEN_AWARE_PARSIMONY,
    # MODES.MODE_MI_LOOKBACK,
    # MODES.MODE_MI_FEUDAL_SYNC_NO_STATE,
    # MODES.MODE_MARGINAL_PARSIMONY,
])
vg.add("bottleneck_coeff", [0.])
vg.add("step_size", [0.01])
vg.add("bonus_step_size", [0.01])
vg.add("exact_stop_gradient", [True])#, False])
vg.add("bottleneck_nonlinear", [True])#lambda deterministic_bottleneck: [True, False] if deterministic_bottleneck else [
    # False])
vg.add("deterministic_bottleneck", [True])#, False])
vg.add("separate_bottlenecks", [True])#, False])#, False])
vg.add("use_decision_nodes", [True])#, False])#, False])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = PermGridEnv(size=v["grid_size"], n_objects=v["grid_size"], object_seed=0)
    policy = StochasticGRUPolicy(
        env_spec=env.spec,
        n_subgoals=v["grid_size"],
        action_bottleneck_dim=5,
        hidden_bottleneck_dim=6,
        bottleneck_dim=7,
        use_bottleneck=True,
        deterministic_bottleneck=v["deterministic_bottleneck"],
        bottleneck_nonlinear=v["bottleneck_nonlinear"],
        separate_bottlenecks=v["separate_bottlenecks"],
        use_decision_nodes=v["use_decision_nodes"]
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_baseline = LinearFeatureBaseline(env_spec=env.spec)

    bonus_evaluator = DiscreteBonusEvaluator(
        env_spec=env.spec,
        policy=policy,
        mode=v["mode"],
        bonus_coeff=1.0,
        bottleneck_coeff=v["bottleneck_coeff"],
        regressor_args=dict(
            use_trust_region=False,
            step_size=0.01,
        ),
        use_exact_regressor=True,
        exact_stop_gradient=v["exact_stop_gradient"],
        exact_entropy=False,
    )
    algo = AltBonusTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_baseline=bonus_baseline,
        bonus_evaluator=bonus_evaluator,
        batch_size=v["batch_size"],
        step_size=v["step_size"],#0.01,
        bonus_step_size=v["bonus_step_size"],#0.005,
        max_path_length=100,
        n_itr=500,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="hrl_sep_bottleneck_2",
        n_parallel=2,
        seed=v["seed"],
        mode="local"
        # env=dict(THEANO_FLAGS="optimizer=None,mode=FAST_COMPILE")
    )
    # sys.exit()
