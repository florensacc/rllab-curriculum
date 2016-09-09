


from sandbox.rocky.hrl_new.algos.hrl_algos1 import HierPolopt
from sandbox.rocky.hrl.envs.atari import AtariEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.hrl_new.policies.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.straw.optimizers.tf_conjugate_gradient_optimizer import ConjugateGradientOptimizer, \
    FiniteDifferenceHvp, PerlmutterHvp
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator

"""
Test new algorithm procedure
"""

vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(5)])
vg.add("game", ["seaquest", "breakout", "frostbite", "space_invaders", "freeway"])
vg.add("algo", [TRPO, HierPolopt])
vg.add("subgoal_interval", [1, 3, 10])
variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:

    env = TfEnv(AtariEnv(game=v["game"], obs_type="ram", frame_skip=4))

    policy = FixedClockPolicy(
        env_spec=env.spec,
        subgoal_dim=50,
        bottleneck_dim=50,
        subgoal_interval=v["subgoal_interval"],
        hidden_sizes=(32, 32),
        log_prob_tensor_std=1.0,
        name="policy"
    )
    aux_policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        name="aux_policy"
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = v["algo"](
        env=env,
        policy=policy,
        aux_policy=aux_policy,
        baseline=baseline,
        max_path_length=4500,
        batch_size=50000,
        n_itr=500,
        discount=0.99,
        gae_lambda=0.99,
        mi_coeff=0.,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="0709-hrl-atari-ram-12",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        mode="lab_kube",
        variant=v,
    )
