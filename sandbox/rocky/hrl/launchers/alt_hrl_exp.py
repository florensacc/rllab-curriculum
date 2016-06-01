from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from sandbox.rocky.hrl.algos.alt_bonus_algos import AltBonusTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("grid_size", [5])  # , 7, 9, 11])
vg.add("batch_size", [20000])  # , 10000, 20000])
vg.add("seed", [11, 111, 211])#, 311, 411])
vg.add("mode", [
    MODES.MODE_MI_FEUDAL_SYNC,
    MODES.MODE_MI_FEUDAL,
    MODES.MODE_BOTTLENECK_ONLY,
    MODES.MODE_JOINT_MI_PARSIMONY,
    MODES.MODE_MARGINAL_PARSIMONY,
    MODES.MODE_HIDDEN_AWARE_PARSIMONY,
])
vg.add("bottleneck_coeff", [0.1])
vg.add("step_size", [0.1, 0.05, 0.01])
vg.add("bonus_step_size", [0.1, 0.05, 0.01])
vg.add("exact_stop_gradient", [True, False])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = PermGridEnv(size=v["grid_size"], n_objects=v["grid_size"], object_seed=0)
    policy = StochasticGRUPolicy(
        env_spec=env.spec,
        n_subgoals=v["grid_size"],
        bottleneck_dim=5,
        use_bottleneck=True,
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
        exp_prefix="hier_alt_fixed2",
        n_parallel=1,
        seed=v["seed"],
        mode="lab_kube"
        # env=dict(THEANO_FLAGS="optimizer=None,mode=FAST_COMPILE")
    )
