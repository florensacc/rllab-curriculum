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
vg.add("grid_size", [10])
vg.add("order_length", [4])
vg.add("n_training_perm", [40])
vg.add("n_test_perm", [100])
vg.add("batch_size", [20000])
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("mode", [
    MODES.MODE_MI_FEUDAL_SYNC,
])
vg.add("bottleneck_coeff", [0.])
vg.add("step_size", [0.01])
vg.add("bonus_step_size", [0.01, 0.005, 0.001, 0.0001])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = PermGridEnv(
        size=v["grid_size"],
        n_objects=v["grid_size"],
        object_seed=0,
        perm_seed=0,
        n_fixed_perm=v["n_training_perm"],
        order_length=v["order_length"],
    )
    test_env = PermGridEnv(
        size=v["grid_size"],
        n_objects=v["grid_size"],
        object_seed=0,
        perm_seed=1,  # use different permutations for testing
        n_fixed_perm=v["n_test_perm"],
        order_length=v["order_length"],
    )
    policy = StochasticGRUPolicy(
        env_spec=env.spec,
        n_subgoals=v["grid_size"],
        bottleneck_dim=5,
        use_bottleneck=True,
        deterministic_bottleneck=True,
        bottleneck_nonlinear=True,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    bonus_baseline = LinearFeatureBaseline(env_spec=env.spec)

    bonus_evaluator = DiscreteBonusEvaluator(
        env_spec=env.spec,
        policy=policy,
        mode=v["mode"],
        bonus_coeff=1.0,
        bottleneck_coeff=0.,
        regressor_args=dict(
            use_trust_region=False,
            step_size=0,
        ),
        use_exact_regressor=True,
        exact_stop_gradient=False,
        exact_entropy=False,
    )
    algo = AltBonusTRPO(
        env=env,
        test_env=test_env,
        policy=policy,
        baseline=baseline,
        bonus_baseline=bonus_baseline,
        bonus_evaluator=bonus_evaluator,
        batch_size=v["batch_size"],
        n_test_samples=10000,
        test_max_path_length=100,
        step_size=v["step_size"],  # 0.01,
        bonus_step_size=v["bonus_step_size"],  # 0.005,
        max_path_length=100,
        n_itr=500,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="hier_gen",
        n_parallel=3,
        seed=v["seed"],
        mode="lab_kube",
        resources=dict(
            requests=dict(
                cpu=3.3,
            ),
            limits=dict(
                cpu=3.3,
            )
        ),
        node_selector={
            "aws/type": "m4.xlarge",
        }
    )
