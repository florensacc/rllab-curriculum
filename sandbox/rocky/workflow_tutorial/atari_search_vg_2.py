from rllab.algos.trpo import TRPO
from sandbox.rocky.hrl.envs.atari import AtariEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def discount(self):
        return [0.99, 0.999, 1.0]

    @variant
    def gae_lambda(self, discount):
        if discount == 1.0:
            return [0.9, 0.99, 0.995, 1.0]
        return [0.99, 0.995, 1.0]


for v in VG().variants():
    env = AtariEnv(game="seaquest")
    policy = CategoricalMLPPolicy(env_spec=env.spec)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        max_path_length=4500,
        batch_size=50000,
        discount=v["discount"],
        gae_lambda=v["gae_lambda"],
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="atari_tutorial",
        n_parallel=4,
        snapshot_mode="last",
        variant=v,
        mode="lab_kube",
    )
