from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from sandbox.rocky.unn_hrl.algos.trpo_unn import TRPO
from sandbox.rocky.unn_hrl.envs.zero_reward import zero_reward
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator

env = zero_reward(SwimmerEnv())

vg = VariantGenerator()
vg.add("seed", [11, 211, 311, 411, 511])
vg.add("policy", [
    GaussianMLPPolicy(env_spec=env.spec),
    GaussianGRUPolicy(env_spec=env.spec),
])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(hidden_sizes=(32,)),
    )

    batch_size = 5000

    algo = TRPO(
        env=env,
        policy=v["policy"],
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500,
        n_itr=1000,
        step_size=0.01,
        eta=0.001,
        eta_discount=1.0,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_reverse_kl_reg=False,
        use_replay_pool=True,
        # do not normalize
        use_kl_ratio=False,
        use_kl_ratio_q=False,
        n_itr_update=1,
        kl_batch_size=64,
        normalize_reward=False,
        replay_pool_size=100000,
        n_updates_per_sample=500,
        second_order_update=True,
        unn_n_hidden=[32],
        unn_layers_type=[1, 1],
        unn_learning_rate=0.0001
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="unn_hrl",
        n_parallel=3,
        snapshot_mode="last",
        mode="lab_kube",
        seed=v["seed"],
    )
