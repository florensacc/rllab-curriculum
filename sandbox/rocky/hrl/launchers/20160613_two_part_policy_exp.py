



from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.normalized_env import normalize
from sandbox.rocky.hrl.algos.multi_joint_algos import MultiJointTRPO
from sandbox.rocky.hrl.bonus_evaluators.zero_bonus_evaluator import ZeroBonusEvaluator
from sandbox.rocky.hrl.envs.point_grid_env import PointGridEnv
from sandbox.rocky.hrl.policies.two_part_policy.two_part_policy import TwoPartPolicy, DuelTwoPartPolicy
from sandbox.rocky.hrl.policies.two_part_policy.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.deterministic_mlp_policy import DeterministicMLPPolicy
import lasagne.nonlinearities as NL

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

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
        "....",
        "Goo.",
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


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 111, 211, 311, 411]

    @variant
    def speed(self):
        return [0.1, 0.2, 0.3]

    @variant
    def subgoal_dim(self):
        return [2, 5, 10]

    @variant
    def high_policy_cls(self):
        return [DeterministicMLPPolicy, GaussianMLPPolicy]

    @variant
    def high_policy_args(self, high_policy_cls):
        if high_policy_cls == DeterministicMLPPolicy:
            yield dict(
                output_nonlinearity=NL.tanh,
            )
        elif high_policy_cls == GaussianMLPPolicy:
            yield dict(
                adaptive_std=True,
                std_share_network=False,
                output_nonlinearity=NL.tanh,
            )
        else:
            raise NotImplementedError

    @variant
    def reparametrize_high_actions(self, high_policy_cls):
        if high_policy_cls == DeterministicMLPPolicy:
            return [True]
        elif high_policy_cls == GaussianMLPPolicy:
            return [True, False]
        else:
            raise NotImplementedError


variants = VG().variants(randomized=True)
print("#Experiments: %d" % len(variants))

for v in variants:

    envs = [normalize(PointGridEnv(desc=map_, speed=v["speed"])) for map_ in MAPS]
    master_policy = TwoPartPolicy(
        env_spec=envs[0].spec,
        subgoal_dim=v["subgoal_dim"],
        high_policy_cls=v["high_policy_cls"],
        high_policy_args=v["high_policy_args"],
        low_policy_cls=GaussianMLPPolicy,
        low_policy_args=dict(
            adaptive_std=True,
            std_share_network=False,
        ),
    )
    policies = []
    baselines = []
    bonus_evaluators = []

    for env in envs:
        policy = DuelTwoPartPolicy(
            env_spec=env.spec,
            master_policy=master_policy,
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
        exp_prefix="two_part_policy_exp_2",
        snapshot_mode="last",
        n_parallel=3,
        seed=v["seed"],
        mode="lab_kube",
        variant=v,
    )
