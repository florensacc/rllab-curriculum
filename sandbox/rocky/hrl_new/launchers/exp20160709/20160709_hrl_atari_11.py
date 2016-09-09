


from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.hrl_new.algos.hrl_algos import HierTRPO
from sandbox.rocky.hrl.envs.atari import AtariEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.policies.fixed_clock_policy1 import FixedClockPolicy as FixedClockPolicy1
from sandbox.rocky.hrl_new.policies.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.straw.optimizers.tf_conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator

"""
Run sanity check with perlmutter
"""

vg = VariantGenerator()
vg.add("policy", ["mlp", "hrl_mlp_fixed", "hrl_mlp"])
vg.add("seed", [x * 100 + 11 for x in range(5)])
vg.add("dim_combo",
       lambda policy: [(18, 1), (1, 18)] if policy == "hrl_mlp_fixed" else [(50, 50)] if policy == "hrl_mlp" else [
           None])
vg.add("subgoal_interval", [1])
vg.add("hidden_sizes", [(32, 32)])
vg.add("batch_size", [50000])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = TfEnv(AtariEnv(game="seaquest", obs_type="ram", frame_skip=4))
    if v["policy"] == "mlp":
        policy = CategoricalMLPPolicy(env_spec=env.spec, name="policy")
        continue
    elif v["policy"] == "hrl_mlp_fixed":
        policy = FixedClockPolicy1(
            env_spec=env.spec,
            subgoal_dim=v["dim_combo"][0],
            bottleneck_dim=v["dim_combo"][1],
            subgoal_interval=v["subgoal_interval"],
            hidden_sizes=v["hidden_sizes"],
            # log_prob_tensor_std=v["log_prob_tensor_std"],
            name="policy"
        )
    else:
        policy = FixedClockPolicy(
            env_spec=env.spec,
            subgoal_dim=v["dim_combo"][0],
            bottleneck_dim=v["dim_combo"][1],
            subgoal_interval=v["subgoal_interval"],
            hidden_sizes=v["hidden_sizes"],
            name="policy"
            # log_prob_tensor_std=v["log_prob_tensor_std"],
        )
        continue

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=4500,
        batch_size=v["batch_size"],
        discount=0.99,
        gae_lambda=0.99,
        optimizer=ConjugateGradientOptimizer(),
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="0709-hrl-atari-ram-11",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        variant=v,
        mode="lab_kube",
    )
    # break
