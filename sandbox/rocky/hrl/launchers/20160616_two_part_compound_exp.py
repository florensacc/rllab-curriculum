


from sandbox.rocky.hrl.algos.multi_joint_algos import MultiJointTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.hrl.policies.two_part_policy.two_part_policy import TwoPartPolicy, DuelTwoPartPolicy
from sandbox.rocky.hrl.policies.two_part_policy.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.reflective_stochastic_mlp_policy import ReflectiveStochasticMLPPolicy
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.bonus_evaluators.two_part_bonus_evaluator import TwoPartBonusEvaluator
import sys

stub(globals())

from rllab.misc.instrument import VariantGenerator

N_ITR = 500

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
variants = vg.variants()

print("#Experiments: %d" % len(variants))

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
            obs_include_history=True,
        )
        if 'G' in "".join(map):
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

    master_policy = TwoPartPolicy(
        env_spec=env_configs[0]["env"].spec,
        high_policy_cls=ReflectiveStochasticMLPPolicy,
        high_policy_args=dict(
            action_policy_cls=CategoricalMLPPolicy,
            action_policy_args=dict(),
            gate_policy_cls=DeterministicMLPPolicy,
            gate_policy_args=dict(
                output_nonlinearity=None
            ),
            gated=v["use_decision_nodes"],
        ),
        low_policy_cls=CategoricalMLPPolicy,
        low_policy_args=dict(),
        subgoal_dim=5,
    )

    policies = []
    baselines = []
    bonus_evaluators = []

    for idx, env_config in enumerate(env_configs):
        env_policy = DuelTwoPartPolicy(
            env_spec=env_config["env"].spec,
            master_policy=master_policy,
            share_gate=True
        )
        env_baseline = LinearFeatureBaseline(env_spec=env_config["env"].spec)
        env_bonus_evaluator = TwoPartBonusEvaluator(
            env_spec=env_config["env"].spec,
            policy=env_policy,
            regressor_args=dict(
                use_trust_region=False,
            ),
            exact_stop_gradient=True,
            bonus_coeff=env_config["bonus_coeff"],
        )

        policies.append(env_policy)
        baselines.append(env_baseline)
        bonus_evaluators.append(env_bonus_evaluator)

    algo = MultiJointTRPO(
        envs=[cfg["env"] for cfg in env_configs],
        policies=policies,
        baselines=baselines,
        bonus_evaluators=bonus_evaluators,
        loss_weights=v["loss_weights"],
        kl_weights=v["kl_weights"],
        step_size=0.01,
        batch_size=20000,
        max_path_length=100,
        reward_coeffs=[cfg["reward_coeff"] for cfg in env_configs],
        scopes=[cfg["name"] for cfg in env_configs],
        n_itr=N_ITR,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="two_part_compound_exp",
        n_parallel=3,
        snapshot_mode="last",
        seed=v["seed"],
        mode="lab_kube",
        variant=v,
    )
    # sys.exit()
