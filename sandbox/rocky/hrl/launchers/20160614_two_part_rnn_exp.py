from __future__ import absolute_import
from __future__ import print_function


from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.algos.multi_joint_algos import MultiJointTRPO
from sandbox.rocky.hrl.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.hrl.bonus_evaluators.zero_bonus_evaluator import ZeroBonusEvaluator
from sandbox.rocky.hrl.envs.point_grid_env import PointGridEnv
from sandbox.rocky.hrl.policies.two_part_policy.two_part_policy import TwoPartPolicy, DuelTwoPartPolicy
from sandbox.rocky.hrl.policies.two_part_policy.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.deterministic_mlp_policy import DeterministicMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.reflective_deterministic_mlp_policy import ReflectiveDeterministicMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.reflective_stochastic_mlp_policy import ReflectiveStochasticMLPPolicy
import lasagne.nonlinearities as NL
import sys

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant

vg = VariantGenerator()

MAPS = [
    # [
    #     "SG..",
    #     "....",
    #     "....",
    #     "....",
    # ],
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
        return [100*x+11 for x in range(10)]

    @variant
    def speed(self):
        return [0.1]

    @variant
    def subgoal_dim(self):
        return [2, 5, 10]

    @variant
    def high_policy_cls(self):
        return [DeterministicMLPPolicy, GaussianMLPPolicy, ReflectiveStochasticMLPPolicy]
        # return [ReflectiveStochasticMLPPolicy]

    @variant
    def gated(self, high_policy_cls):
        if high_policy_cls == ReflectiveStochasticMLPPolicy:
            return [True, False]
        else:
            return [None]

    @variant
    def share_gate(self, high_policy_cls, gated):
        if high_policy_cls == ReflectiveStochasticMLPPolicy and gated:
            return [True, False]
        else:
            return [True]

    @variant
    def reflective_stochastic(self, high_policy_cls):
        # yield True
        if high_policy_cls == ReflectiveStochasticMLPPolicy:
            return [True, False]
        else:
            return [None]

    @variant
    def high_policy_args(self, high_policy_cls, gated, reflective_stochastic):
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
        elif high_policy_cls == ReflectiveStochasticMLPPolicy:
            if reflective_stochastic:
                policy_cls = GaussianMLPPolicy
            else:
                policy_cls = DeterministicMLPPolicy
            yield dict(
                action_policy_cls=policy_cls,
                action_policy_args=dict(
                    output_nonlinearity=NL.tanh,
                ),
                gate_policy_cls=policy_cls,
                gate_policy_args=dict(
                    output_nonlinearity=None,
                ),
                gated=gated,
                # truncate_gradient=10,
            )
        else:
            raise NotImplementedError

    @variant
    def reparametrize_high_actions(self, high_policy_cls):
        yield True
        # if high_policy_cls == DeterministicMLPPolicy:
        #     return [True]
        # elif high_policy_cls == GaussianMLPPolicy:
        #     return [True, False]
        # elif high_policy_cls == ReflectiveStochasticMLPPolicy:
        #     return [True, False]
        # else:
        #     raise NotImplementedError


variants = VG().variants()#randomized=True)
print("#Experiments: %d" % len(variants))

for v in variants:

    envs = [PointGridEnv(desc=map_, speed=v["speed"]) for map_ in MAPS]
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
        reparametrize_high_actions=v["reparametrize_high_actions"]
    )
    policies = []
    baselines = []
    bonus_evaluators = []

    for env in envs:
        policy = DuelTwoPartPolicy(
            env_spec=env.spec,
            master_policy=master_policy,
            share_gate=v["share_gate"],
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
        optimizer=ConjugateGradientOptimizer(
            hvp_approach=FiniteDifferenceHvp(grad_clip=10),
            subsample_factor=.1,
        )
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="two_part_rnn_exp",
        snapshot_mode="last",
        n_parallel=3,
        seed=v["seed"],
        mode="lab_kube",
        variant=v,
        # env=dict(THEANO_FLAGS="mode=FAST_COMPILE"),
    )

    # sys.exit()
