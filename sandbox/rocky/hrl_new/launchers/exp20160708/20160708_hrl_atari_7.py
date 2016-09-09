


from sandbox.rocky.hrl_new.algos.hrl_algos import HierTRPO
from sandbox.rocky.hrl.envs.atari import AtariEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.hrl_new.policies.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

stub(globals())
from rllab.misc.instrument import VariantGenerator

"""
Add back baseline
"""

vg = VariantGenerator()
vg.add("seed", [x * 100 + 11 for x in range(5)])
vg.add("subgoal_dim", [50])#10, 50])#], 50])
vg.add("bottleneck_dim", [50])#10, 50])
vg.add("subgoal_interval", [1])#1, 3, 10])
vg.add("mi_coeff", [0.])#, 0.01, 0.1, 1., 10.])
vg.add("hidden_sizes", [(32, 32)])#, (64, 64), (256, 256)])
vg.add("batch_size", [50000])
vg.add("max_opt_itr", [20])#, 50, 100])
vg.add("log_prob_tensor_std", [1.0, 10.0, 0.1, 0.01])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = TfEnv(AtariEnv(game="seaquest", obs_type="ram", frame_skip=4))
    policy = FixedClockPolicy(
        env_spec=env.spec,
        subgoal_dim=v["subgoal_dim"],
        bottleneck_dim=v["bottleneck_dim"],
        subgoal_interval=v["subgoal_interval"],
        hidden_sizes=v["hidden_sizes"],
        log_prob_tensor_std=v["log_prob_tensor_std"],
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = HierTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=4500,
        batch_size=v["batch_size"],
        discount=0.99,
        gae_lambda=0.99,
        mi_coeff=v["mi_coeff"],
        optimizer=PenaltyLbfgsOptimizer(name="optimizer", max_opt_itr=v["max_opt_itr"]),
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="0708-hrl-atari-ram-7",
        seed=v["seed"],
        n_parallel=4,
        snapshot_mode="last",
        variant=v,
        mode="lab_kube",
    )
    # break
