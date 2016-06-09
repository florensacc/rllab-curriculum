from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.algos.duel_algos import DuelBatchPolopt
from sandbox.rocky.hrl.algos.bonus_algos import BonusTRPO
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.hrl.policies.duel_stochastic_gru_policy import DuelStochasticGRUPolicy
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.misc.instrument import stub, run_experiment_lite
# from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES
from sandbox.rocky.hrl.bonus_evaluators.duel_discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES

stub(globals())

from rllab.misc.instrument import VariantGenerator

max_paths_lengths = {5: 100, 7: 200, 9: 300}
vg = VariantGenerator()
vg.add("grid_size", [5, 7, 9])
vg.add("max_path_length", lambda grid_size: [max_paths_lengths[grid_size]])
vg.add("bonus_step_size", [0.1, 0.01, 0.001, 0.0001, 1e-6])
vg.add("use_decision_nodes", [True, False])
vg.add("seed", [11, 111, 211, 311, 411])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    # env = PermGridEnv(size=v["grid_size"], n_objects=v["grid_size"], object_seed=0)
    wrapped_env = GridWorldEnv(
        desc=[
            "SFFF",
            "FFFF",
            "FFFF",
            "FFFG"
        ]
    )
    env = CompoundActionSequenceEnv(
        wrapped_env=wrapped_env,
        action_map=[
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        obs_include_history=True,
    )

    policy = DuelStochasticGRUPolicy(
        env_spec=env.spec, n_subgoals=v["grid_size"], use_decision_nodes=v["use_decision_nodes"])
    master_baseline = LinearFeatureBaseline(env_spec=env.spec)
    duel_baseline = LinearFeatureBaseline(env_spec=env.spec)

    bonus_evaluator_args = dict(
        env_spec=env.spec,
        mode=MODES.MODE_MI_FEUDAL_SYNC,
        bottleneck_coeff=0.,
        regressor_args=dict(
            use_trust_region=False,
            step_size=0.01,
        ),
        use_exact_regressor=True,
        exact_stop_gradient=True,
        exact_entropy=False,
    )

    master_bonus_evaluator = DiscreteBonusEvaluator(
        bonus_coeff=0.0,
        policy=policy,
        **bonus_evaluator_args
    )

    duel_bonus_evaluator = DiscreteBonusEvaluator(
        bonus_coeff=1.0,
        policy=policy.duel_policy,
        **bonus_evaluator_args
    )

    algo = DuelBatchPolopt(
        env=env,
        policy=policy,
        baseline=master_baseline,
        master_algo=BonusTRPO(
            env=env,
            policy=policy,
            baseline=master_baseline,
            bonus_evaluator=master_bonus_evaluator,
            scope="master_algo",
            batch_size=20000,
            max_path_length=v["max_path_length"],
            step_size=0.01,
        ),
        duel_algo=BonusTRPO(
            env=env,
            policy=policy.duel_policy,
            baseline=duel_baseline,
            reward_coeff=0.,
            bonus_evaluator=duel_bonus_evaluator,
            scope="duel_algo",
            batch_size=20000,
            max_path_length=v["max_path_length"],
            step_size=v["bonus_step_size"],
        )
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="duel_hrl_exp_2",
        n_parallel=3,
        # resources=dict(
        #     requests=dict(
        #         cpu=3.3,
        #     ),
        #     limits=dict(
        #         cpu=3.3,
        #     )
        # ),
        # node_selector={
        #     "aws/type": "m4.xlarge"
        # },
        seed=v["seed"],
        mode="lab_kube"
    )
