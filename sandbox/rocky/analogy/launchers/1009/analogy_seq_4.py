import os

from sandbox.rocky.analogy.utils import conopt_run_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale

from rllab.misc.instrument import VariantGenerator, variant, run_experiment_lite

"""
attention
"""

USE_GPU = True # True
USE_CIRRASCALE = True # True
# MODE = "local_docker"  # launch_cirrascale#(##)"local_docker"
# MODE = "local_docker"
MODE = launch_cirrascale#()
SETUP = "simple"
ENV = "seq"
VERSION = "v3"


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def n_train_trajs(self):
        return [100000]  # 10000, 50000, 100000]

    @variant
    def use_shuffler(self):
        return [True]

    @variant
    def state_include_action(self):
        return [False]

    @variant
    def horizon(self):
        return [200]

    @variant
    def n_epochs(self):
        return [10000]

    @variant
    def obs_type(self):
        return ["full_state"]

    @variant
    def obs_size(self):
        yield (30, 30)

    @variant
    def max_seq_length(self):
        yield 20#1#20  # 6#1#6#1

    @variant
    def min_seq_length(self):
        yield 1
        # yield 3

    @variant
    def n_particles(self):
        return [3]#, 4, 5, 6]
        # yield 3#4#6  # 2
        # yield 4

    @variant
    def batch_size(self):
        return [128]#128]#, 256, 512]#1024]

    # @variant(hide=True)
    # def demo_cache_key(self, obs_type, min_seq_length, max_seq_length, n_particles, n_train_trajs):
    #     yield "-".join([
    #         SETUP,
    #         ENV,
    #         str(horizon),
    #         obs_type.replace("_", "-"),
    #         VERSION,
    #         "minseq" + str(min_seq_length),
    #         "maxseq" + str(max_seq_length),
    #         "pts" + str(n_particles),
    #         "n" + str(n_train_trajs)
    #     ])


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))


def run_task(v):
    from sandbox.rocky.analogy.algos.trainer import Trainer
    from sandbox.rocky.analogy.demo_collector.policy_demo_collector import PolicyDemoCollector

    from sandbox.rocky.analogy.envs.simple_particle_env import SimpleParticleEnv
    from sandbox.rocky.analogy.policies.simple_particle_tracking_policy import SimpleParticleTrackingPolicy

    from sandbox.rocky.analogy.policies.modular_analogy_policy import ModularAnalogyPolicy
    # from sandbox.rocky.analogy.networks.simple_particle.double_rnn import Net
    from sandbox.rocky.analogy.networks.simple_particle.attention import Net
    from sandbox.rocky.tf.envs.base import TfEnv

    env_args = dict(
        n_particles=v["n_particles"],
        max_seq_length=v["max_seq_length"],
        min_seq_length=v["min_seq_length"],
        obs_type=v["obs_type"],
        obs_size=v["obs_size"]
    )

    env = TfEnv(SimpleParticleEnv(
        seed=0,
        **env_args
    ))

    policy = ModularAnalogyPolicy(
        env_spec=env.spec,
        name="policy",
        net=Net(obs_type=v["obs_type"])
    )

    def demo_cache_key(trainer, env, policy):
        return "-".join([
            SETUP,
            ENV,
            env.obs_type.replace("_", "-"),
            VERSION,
            "minseq" + str(env.min_seq_length),
            "maxseq" + str(env.max_seq_length),
            "pts" + str(env.n_particles),
            "n" + str(trainer.n_train_trajs)
        ])

    curriculum_env_cls_list = []
    for max_seq_length in range(1, v["max_seq_length"] + 1):
        curriculum_env_cls_list.append(
            TfEnv.wrap(
                SimpleParticleEnv,
                **dict(env_args, min_seq_length=max(1, max_seq_length - 3), max_seq_length=max_seq_length)
            )
        )

    def curriculum_criterion(paths, envs):
        return envs[0].success_rate(paths, envs) >= 0.95

    algo = Trainer(
        policy=policy,
        env_cls=TfEnv.wrap(
            SimpleParticleEnv,
            **env_args
        ),
        demo_collector=PolicyDemoCollector(
            policy_cls=SimpleParticleTrackingPolicy
        ),
        skip_eval=False,
        demo_cache_key=demo_cache_key,
        n_train_trajs=v["n_train_trajs"],
        n_test_trajs=10 if MODE == "local_docker" else 50,
        n_passes_per_epoch=1,
        horizon=v["horizon"],
        n_epochs=v["n_epochs"],
        learning_rate=1e-2,
        no_improvement_tolerance=10,
        shuffler=SimpleParticleEnv.shuffler() if v["use_shuffler"] else None,
        batch_size=v["batch_size"],
        use_curriculum=False,#True,
        curriculum_env_cls_list=curriculum_env_cls_list,
        curriculum_criterion=curriculum_criterion,
    )

    algo.train()


for v in variants:
    conopt_run_experiment(
    # run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="analogy-seq-4",
        mode=MODE,
        n_parallel=8,
        seed=v["seed"],
        snapshot_mode="last",
        variant=v,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        use_gpu=USE_GPU,
    )
    # sys.exit()
