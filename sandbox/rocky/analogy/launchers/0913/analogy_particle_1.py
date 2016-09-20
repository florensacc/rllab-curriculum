from sandbox.rocky.analogy.envs.simple_particle_env import SimpleParticleEnv
from sandbox.rocky.analogy.policies.simple_particle_tracking_policy import SimpleParticleTrackingPolicy
from sandbox.rocky.analogy.policies.double_lstm_policy import DoubleLSTMPolicy
from sandbox.rocky.analogy.policies.demo_gru_mlp_analogy_policy import DemoRNNMLPAnalogyPolicy
from sandbox.rocky.analogy.policies.mlp_analogy_policy import MLPAnalogyPolicy
from sandbox.rocky.analogy.algos.trainer import Trainer
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def n_particles(self):
        return [2, 3, 4]

    @variant
    def n_train_trajs(self):
        return [100, 1000, 10000, 50000]

    @variant
    def hidden_dim(self):
        return [25, 50, 100]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = TfEnv(SimpleParticleEnv(seed=0, n_particles=v["n_particles"]))
    policy = DemoRNNMLPAnalogyPolicy(env_spec=env.spec, name="policy", hidden_size=v["hidden_dim"])
    algo = Trainer(
        policy=policy,
        env_cls=TfEnv.wrap(SimpleParticleEnv, n_particles=v["n_particles"]),
        demo_policy_cls=SimpleParticleTrackingPolicy,
        n_train_trajs=v["n_train_trajs"],
        n_test_trajs=50,
        horizon=20,
        n_epochs=1000,
        learning_rate=1e-2,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="analogy-particle-1",
        mode="lab_kube",
        n_parallel=4,
        seed=v["seed"],
        variant=v,
        snapshot_mode="last",
    )
