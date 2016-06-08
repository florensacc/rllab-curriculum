from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES
from sandbox.rocky.hrl.algos.bonus_algos import BonusTRPO
from sandbox.rocky.hrl.algos.alt_bonus_algos import AltBonusTRPO
from sandbox.rocky.hrl.envs.atari import AtariEnv

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("n_subgoals", [5])
vg.add("batch_size", [10000])
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("bonus_coeff", [1.])
vg.add("bottleneck_coeff", [0.])
vg.add("use_decision_nodes", [False])
vg.add("use_bottleneck", [True])
vg.add("deterministic_bottleneck", [True])
vg.add("bottleneck_nonlinear", [True, False])
vg.add("random_reset", [False])
vg.add("hidden_sizes", [(32, 32)])
vg.add("bottleneck_dim", [10, 20])
vg.add("bonus_step_size", [0.001, 0.005, 0.01])
vg.add("use_exact_regressor", [True])
vg.add("exact_entropy", [False])
vg.add("mode", [
    # MODES.MODE_MARGINAL_PARSIMONY,
    # MODES.MODE_JOINT_MI_PARSIMONY,
    MODES.MODE_MI_FEUDAL_SYNC,
    # MODES.MODE_MI_FEUDAL,
    # MODES.MODE_HIDDEN_AWARE_PARSIMONY,
    # MODES.MODE_BOTTLENECK_ONLY,
])

variants = vg.variants()
print("#Experiments:", len(variants))

for v in variants:
    env = AtariEnv(game="pong", obs_type="ram", frame_skip=4)
    policy = StochasticGRUPolicy(
        env_spec=env.spec,
        n_subgoals=v["n_subgoals"],
        use_decision_nodes=v["use_decision_nodes"],
        random_reset=v["random_reset"],
        use_bottleneck=v["use_bottleneck"],
        bottleneck_dim=v["bottleneck_dim"],
        deterministic_bottleneck=v["deterministic_bottleneck"],
        hidden_sizes=v["hidden_sizes"],
        bottleneck_nonlinear=v["bottleneck_nonlinear"],
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_evaluator = DiscreteBonusEvaluator(
        env_spec=env.spec,
        policy=policy,
        mode=v["mode"],
        bonus_coeff=v["bonus_coeff"],
        bottleneck_coeff=v["bottleneck_coeff"],
        regressor_args=dict(
            use_trust_region=False,
            step_size=0.,
        ),
        use_exact_regressor=v["use_exact_regressor"],
        exact_entropy=v["exact_entropy"],
    )
    algo = AltBonusTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        bonus_baseline=bonus_baseline,
        batch_size=v["batch_size"],
        bonus_evaluator=bonus_evaluator,
        step_size=0.01,
        bonus_step_size=v["bonus_step_size"],
        max_path_length=4500,
        n_itr=500,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="hrl_atari2_bonus",
        n_parallel=2,
        seed=v["seed"],
        mode="lab_kube",
    )
