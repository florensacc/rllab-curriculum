


from rllab.algos.npo import NPO
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.pchen.diag_npg.optimizers.diagonal_natural_gradient_optimizer import DiagonalNaturalGradientOptimizer
from sandbox.pchen.nono.ltrpo import LTRPO
from sandbox.pchen.nono.optimizers.cg_with_history import ConjugateGradientOptimizerWithHistory

from sandbox.pchen.poga.algos.poga import POGA
stub(globals())

env = normalize(
    # CartpoleEnv()
    SwimmerEnv()
)

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [123, 42, 998, 101, 33334444, 3434, 9998887766]

    @variant
    def history_size(self):
        return [200, 400, 1000, 5000, 500000]

    @variant
    def ncg(self):
        return [10, ]

    @variant
    def boh(self):
        return [True, False]


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    # baseline = LinearFeatureBaseline(env_spec=env.spec)
    baseline = ZeroBaseline(env_spec=env.spec)

    budget = 500000
    bs = 200
    algo = LTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=bs,
        max_path_length=100,
        n_itr=min([budget/bs, 200]),
        discount=0.99,
        step_size=0.001,
        # plot=True,
        optimizer=ConjugateGradientOptimizerWithHistory(
            cg_iters=v.ncg,
            reg_coeff=1e-2,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False,
            history_size=v.history_size,
            backtrack_on_history=v.boh,
        ),
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=1,
        snapshot_mode="last",
        seed=v.seed,
        # mode="local",
        mode="lab_kube",
        # plot=True,
        exp_prefix="sbs_history_corrected",
        # resources=dict(
        #     requests=dict(
        #         cpu=1.1,
        #     ),
        #     limits=dict(
        #         cpu=1.1,
        #     )
        # )
    )

