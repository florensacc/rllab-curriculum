


from sandbox.rocky.hrl.envs.point_grid_env import PointGridEnv
from rllab.algos.trpo import TRPO
from sandbox.rocky.hrl.algos.multi_joint_algos import MultiJointTRPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.hrl.policies.two_part_gaussian_mlp_policy import TwoPartGaussianMLPPolicy, \
    DuelTwoPartGaussianMLPPolicy
from sandbox.rocky.hrl.bonus_evaluators.zero_bonus_evaluator import ZeroBonusEvaluator
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import sys

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()

MAPS = [
    [
        "So..",
        ".G..",
        ".oo.",
        "....",
    ],
    [
        "So..",
        ".G..",
        ".oo.",
        "....",
    ],
    [
        "So..",
        "....",
        ".oo.",
        ".G..",
    ],
    [
        "So..",
        "....",
        ".oo.",
        "...G",
    ],
]

# vg.add("map", MAPS)
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("speed", [0.1])  # , 0.05, 0.2, 0.3])
vg.add("subgoal_dim", [2, 5, 10])  # , 0.05, 0.2, 0.3])
vg.add("adaptive_std", [True, False])
vg.add("std_share_network", [True, False])
vg.add("share_std_layers", [True, False])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    envs = [PointGridEnv(desc=map_, speed=v["speed"]) for map_ in MAPS]
    master_policy = TwoPartGaussianMLPPolicy(
        env_spec=envs[0].spec,
        subgoal_dim=v["subgoal_dim"],
        adaptive_std=v["adaptive_std"],
        action_hidden_sizes=(32, 32),
        std_hidden_sizes=(32, 32),
        subgoal_hidden_sizes=(32, 32),
        std_share_network=v["std_share_network"],
    )
    policies = []
    baselines = []
    bonus_evaluators = []

    for env in envs:
        policy = DuelTwoPartGaussianMLPPolicy(
            env_spec=env.spec,
            master_policy=master_policy,
            share_std_layers=v["share_std_layers"]
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        bonus_evaluator = ZeroBonusEvaluator(env_spec=env.spec, policy=policy)

        policies.append(policy)
        baselines.append(baseline)
        bonus_evaluators.append(bonus_evaluator)

    algo = MultiJointTRPO(
        envs=envs,
        policies=policies,
        baselines=baselines,
        loss_weights=[1.] * len(envs),
        kl_weights=[1.] * len(envs),
        reward_coeffs=[1.] * len(envs),
        bonus_evaluators=bonus_evaluators,
        scopes=["Master_%d" % idx for idx in range(len(envs))],
        batch_size=20000,
        max_path_length=500,
        n_itr=500,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="two_part_mlp_exp",
        snapshot_mode="last",
        n_parallel=3,
        seed=v["seed"],
        mode="lab_kube",
        variant=v,
    )
