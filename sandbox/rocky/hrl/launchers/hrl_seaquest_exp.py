from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.hrl.algos.alt_bonus_algos import AltBonusTRPO
from sandbox.rocky.hrl.algos.bonus_algos import BonusTRPO
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.seaquest_grid_world_env import SeaquestGridWorldEnv
from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES
from sandbox.rocky.hrl.misc.control_flows import MultiAlgo
import sys

stub(globals())

from rllab.misc.instrument import VariantGenerator

EXP_MODE = "local"

POLICY_MODE = "baseline"

if POLICY_MODE == "baseline":

    vg = VariantGenerator()
    vg.add("batch_size", [20000])
    vg.add("seed", [11, 111, 211, 311, 411])
    vg.add("step_size", [0.01])
    vg.add("env_size", [6, 7, 8, 10])

    variants = vg.variants()

    print("#Experiments: %d" % len(variants))

    for v in variants:
        envs = [
            SeaquestGridWorldEnv(
                size=v["env_size"], n_bombs=n_divers, n_divers=n_divers,
                fixed_diver_slots=v["env_size"], diver_pos_seed=0
            ) for n_divers in range(1, 4)
            ]

        policy = CategoricalMLPPolicy(
            env_spec=envs[0].spec,
            # hidden_sizes=v["hidden_sizes"],
        )
        # policy = StochasticGRUPolicy(
        #     env_spec=envs[0].spec,
        #     n_subgoals=5,
        #     action_bottleneck_dim=5,
        #     hidden_bottleneck_dim=5,
        #     bottleneck_dim=5,
        #     use_bottleneck=True,
        #     deterministic_bottleneck=True,  # v["deterministic_bottleneck"],
        #     bottleneck_nonlinear=True,  # v["bottleneck_nonlinear"],
        #     separate_bottlenecks=True,  # v["separate_bottlenecks"],
        #     use_decision_nodes=False,  # v["use_decision_nodes"]
        # )
        baseline = LinearFeatureBaseline(env_spec=envs[0].spec)

        algos = []
        for idx, env in enumerate(envs):
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                step_size=v["step_size"],
                max_path_length=100 * (idx + 1),
                n_itr=10,
                batch_size=v["batch_size"],
                plot=True,
            )
            algos.append(algo)

        multi_algo = MultiAlgo(algos_dict=dict(zip(range(len(algos)), algos)))

        run_experiment_lite(
            multi_algo.train(),
            exp_prefix="hrl_seaquest_cur_8",
            n_parallel=4 if EXP_MODE == "local" else 1,
            seed=v["seed"],
            mode=EXP_MODE,
            plot=True,
        )

else:

    vg = VariantGenerator()
    vg.add("batch_size", [20000] if EXP_MODE == "local" else [20000])
    vg.add("seed", [11, 111, 211, 311, 411])
    vg.add("mode", [
        MODES.MODE_MI_FEUDAL_SYNC,
        MODES.MODE_MI_FEUDAL_SYNC_ENT_BONUS,
    ])
    vg.add("bottleneck_coeff", [0.])
    vg.add("step_size", [0.01])
    vg.add("bonus_step_size", [0.01, 0.001, 0.0001, 1e-6])
    vg.add("bonus_coeff", [1.0])  # , 0.1, 0.01, 0.001, 0.0001])
    # vg.add("bonus_coeff", [1.0, 0.1, 0.01, 0.001, 0.0001])
    vg.add("exact_stop_gradient", [True])
    vg.add("bottleneck_nonlinear", [True])
    vg.add("deterministic_bottleneck", [True])
    vg.add("separate_bottlenecks", [True])
    vg.add("use_decision_nodes", [False])

    variants = vg.variants()

    print("#Experiments: %d" % len(variants))

    for v in variants:
        envs = [SeaquestGridWorldEnv(size=5, n_bombs=1, n_divers=n_divers) for n_divers in range(1, 6)]

        policy = StochasticGRUPolicy(
            env_spec=envs[0].spec,
            n_subgoals=5,
            action_bottleneck_dim=5,
            hidden_bottleneck_dim=5,
            bottleneck_dim=5,
            use_bottleneck=True,
            deterministic_bottleneck=v["deterministic_bottleneck"],
            bottleneck_nonlinear=v["bottleneck_nonlinear"],
            separate_bottlenecks=v["separate_bottlenecks"],
            use_decision_nodes=v["use_decision_nodes"]
        )
        baseline = LinearFeatureBaseline(env_spec=envs[0].spec)

        bonus_baseline = LinearFeatureBaseline(env_spec=envs[0].spec)

        bonus_evaluator = DiscreteBonusEvaluator(
            env_spec=envs[0].spec,
            policy=policy,
            mode=v["mode"],
            bonus_coeff=v["bonus_coeff"],
            bottleneck_coeff=v["bottleneck_coeff"],
            regressor_args=dict(
                use_trust_region=False,
                step_size=0.01,

            ),
            use_exact_regressor=True,
            exact_stop_gradient=v["exact_stop_gradient"],
            exact_entropy=False,
        )
        algos = []
        for env in envs:
            algo = AltBonusTRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                bonus_baseline=bonus_baseline,
                bonus_evaluator=bonus_evaluator,
                batch_size=v["batch_size"],
                step_size=v["step_size"],
                bonus_step_size=v["bonus_step_size"],
                max_path_length=100,
                n_itr=100,
            )
            algos.append(algo)

        multi_algo = MultiAlgo(algos_dict=dict(zip(range(len(algos)), algos)))

        run_experiment_lite(
            multi_algo.train(),
            exp_prefix="hrl_seaquest_cur_5",
            n_parallel=4 if EXP_MODE == "local" else 1,
            seed=v["seed"],
            mode=EXP_MODE
        )
