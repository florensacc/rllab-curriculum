from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.algos.multi_alt_batch_polopt import MultiAltBatchPolopt
from sandbox.rocky.hrl.algos.multi_joint_algos import MultiJointTRPO
from sandbox.rocky.hrl.algos.bonus_algos import BonusTRPO
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.hrl.policies.duel_stochastic_gru_policy import DuelStochasticGRUPolicy
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from sandbox.rocky.hrl.envs.zero_reward import zero_reward
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.misc.instrument import stub, run_experiment_lite
# from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES
from sandbox.rocky.hrl.bonus_evaluators.duel_discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES

stub(globals())

from rllab.misc.instrument import VariantGenerator

N_ITR = 500

# algo_cls = MultiJointTRPO

vg = VariantGenerator()
vg.add("use_decision_nodes", [True, False])
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("algo_cls", [MultiJointTRPO])
vg.add("bonus_loss_weight", [0.1, 0.01, 0.001, 0.])
vg.add("weights", lambda bonus_loss_weight: [
    dict(loss_weights=[1., 1., 1., 1., bonus_loss_weight], kl_weights=[1., 1., 1., 1., int(bonus_loss_weight > 0)]),
    dict(loss_weights=[0., 1., 1., 1., bonus_loss_weight], kl_weights=[0., 1., 1., 1., int(bonus_loss_weight > 0)]),
    dict(loss_weights=[0., 0., 1., 1., bonus_loss_weight], kl_weights=[0., 0., 1., 1., int(bonus_loss_weight > 0)]),
    dict(loss_weights=[0., 0., 0., 1., bonus_loss_weight], kl_weights=[0., 0., 0., 1., int(bonus_loss_weight > 0)]),
])
vg.add("loss_weights", lambda weights: [weights["loss_weights"]])
vg.add("kl_weights", lambda weights: [weights["kl_weights"]])
# vg.add("bonus_step_size", [0.01])
variants = vg.variants()

print("#Experiments: %d" % len(variants))

# MAPS =

for v in variants:
    maps = [
        [
            "S....",
            ".G...",
            ".....",
            ".....",
            ".....",
        ],
        [
            "S....",
            ".o...",
            "..G..",
            ".....",
            ".....",
        ],
        [
            "S....",
            ".o...",
            "..o..",
            "...G.",
            ".....",
        ],
        [
            "S....",
            ".o.o.",
            ".....",
            "...oo",
            "....G",
        ],
        [
            "S....",
            ".....",
            ".....",
            ".....",
            ".....",
        ],
    ]
    env_configs = []
    for idx, map in enumerate(maps):
        wrapped_env = GridWorldEnv(desc=map)
        env = CompoundActionSequenceEnv(
            wrapped_env=wrapped_env,
            action_map=[
                [0, 1, 1],
                [1, 3, 3],
                [2, 2, 0],
                [3, 0, 2],
            ],
            # action_dim=2,
            obs_include_history=True,
        )
        if 'G' in "".join(map):#()#idx < len(maps) - 1:
            env_configs.append(
                dict(
                    env=env, reward_coeff=1.0, bonus_coeff=0.0, name="Master_%d" % idx
                )
            )
        else:
            env_configs.append(
                dict(
                    env=env, reward_coeff=0.0, bonus_coeff=1.0, name="Duel"
                )
            )

    # env_configs = [
    #     dict(
    #         env=env, reward_coeff=1.0, bonus_coeff=0.0, name="Master"
    #     ),
    #     dict(
    #         env=env, reward_coeff=0.0, bonus_coeff=1.0, name="Duel"
    #     )
    # ]

    master_policy = StochasticGRUPolicy(
        env_spec=env_configs[0]["env"].spec,
        n_subgoals=5,
        use_decision_nodes=v["use_decision_nodes"],
        use_bottleneck=False,
    )

    policies = []
    baselines = []
    bonus_evaluators = []

    for idx, env_config in enumerate(env_configs):
        env_policy = DuelStochasticGRUPolicy(
            env_spec=env_config["env"].spec,
            master_policy=master_policy
        )
        env_baseline = LinearFeatureBaseline(env_spec=env_config["env"].spec)

        env_bonus_evaluator = DiscreteBonusEvaluator(
            bonus_coeff=env_config["bonus_coeff"],
            policy=env_policy,
            env_spec=env_config["env"].spec,
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
        policies.append(env_policy)
        baselines.append(env_baseline)
        bonus_evaluators.append(env_bonus_evaluator)

    if v["algo_cls"] == MultiJointTRPO:
        algo = MultiJointTRPO(
            envs=dict(enumerate([cfg["env"] for cfg in env_configs])),
            policies=dict(enumerate(policies)),
            baselines=dict(enumerate(baselines)),
            bonus_evaluators=dict(enumerate(bonus_evaluators)),
            loss_weights=v["loss_weights"],#[cfg["loss_weight"] for cfg in env_configs],
            kl_weights=v["kl_weights"],#[1.] * len(env_configs),
            step_size=0.01,
            batch_size=20000,
            max_path_length=100,
            reward_coeffs=[cfg["reward_coeff"] for cfg in env_configs],
            scopes=[cfg["name"] for cfg in env_configs],
            n_itr=N_ITR,
        )
    elif v["algo_cls"] == MultiAltBatchPolopt:
        raise NotImplementedError
        algos = []
        for idx, (env_config, env_policy, env_baseline, env_bonus_evaluator) in enumerate(zip(
                env_configs, policies, baselines, bonus_evaluators)):
            env_algo = BonusTRPO(
                env=env_config["env"],
                policy=env_policy,
                baseline=env_baseline,
                bonus_evaluator=env_bonus_evaluator,
                reward_coeff=env_config["reward_coeff"],
                scope=env_config["name"],
                batch_size=20000,
                max_path_length=100,
                step_size=0.01 if idx < len(maps) - 1 else v["bonus_step_size"],
            )
            algos.append(dict(algo=env_algo, name=env_config["name"]))
        algo = MultiAltBatchPolopt(
            algos=dict(enumerate(algos)),
            n_itr=N_ITR,
        )
    else:
        raise NotImplementedError

    run_experiment_lite(
        algo.train(),
        exp_prefix="actionseq_multi_hrl_exp_11",
        n_parallel=3,
        snapshot_mode="last",
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
    # sys.exit()
