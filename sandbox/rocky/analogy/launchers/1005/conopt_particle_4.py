import os

from sandbox.rocky.analogy.utils import conopt_run_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale

from rllab.misc.instrument import VariantGenerator, variant

"""
Test performance of new network
"""

USE_GPU = True
USE_CIRRASCALE = True
# MODE = "local_docker"#launch_cirrascale#(##)"local_docker"
MODE = launch_cirrascale
SETUP = "conopt"
ENV = "particle-2"
VERSION = "v8"



class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def n_train_trajs(self):
        return [100, 500, 1000]

    @variant
    def use_shuffler(self):
        return [True]

    @variant
    def state_include_action(self):
        return [False]

    @variant
    def horizon(self):
        return [100]

    @variant
    def n_epochs(self):
        return [1000]

    @variant
    def obs_type(self):
        return ["image"]

    @variant
    def obs_size(self):
        yield (30, 30)

    @variant(hide=True)
    def demo_cache_key(self, horizon, obs_type):
        yield "%s-%s-%s-%s-%s" % (
            SETUP,
            ENV,
            str(horizon),
            obs_type.replace("_", "-"),
            VERSION
        ),


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))


def run_task(v):
    from sandbox.rocky.analogy.algos.trainer import Trainer
    from sandbox.rocky.analogy.demo_collector.policy_demo_collector import PolicyDemoCollector
    from sandbox.rocky.analogy.demo_collector.trajopt_demo_collector import TrajoptDemoCollector

    if SETUP == "simple":
        from sandbox.rocky.analogy.envs.simple_particle_env import SimpleParticleEnv
        from sandbox.rocky.analogy.policies.simple_particle_tracking_policy import SimpleParticleTrackingPolicy
    else:
        from sandbox.rocky.analogy.envs.conopt_particle_env import ConoptParticleEnv
        from sandbox.rocky.analogy.policies.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy

    from sandbox.rocky.analogy.policies.modular_analogy_policy import ModularAnalogyPolicy
    from sandbox.rocky.analogy.networks.conopt_particle.double_rnn import Net
    from sandbox.rocky.tf.envs.base import TfEnv

    if SETUP == "conopt":
        env = TfEnv(ConoptParticleEnv(
            seed=0,
            obs_type=v["obs_type"],
            obs_size=v["obs_size"]
        ))
    else:
        env = TfEnv(SimpleParticleEnv(
            seed=0,
            obs_type=v["obs_type"],
            obs_size=v["obs_size"]
        ))
    # summary_net, action_net = net.new_networks(env.spec)

    policy = ModularAnalogyPolicy(
        env_spec=env.spec,
        name="policy",
        net=Net(obs_type=v["obs_type"]),
    )

    algo = Trainer(
        policy=policy,
        env_cls=TfEnv.wrap(
            ConoptParticleEnv if SETUP == "conopt" else
            SimpleParticleEnv,
            obs_type=v["obs_type"],
            obs_size=v["obs_size"],
        ),
        demo_collector=PolicyDemoCollector(
            policy_cls=ConoptParticleTrackingPolicy if SETUP == "conopt" else
            SimpleParticleTrackingPolicy
        ),
        skip_eval=False,#True,#False,#True,
        demo_cache_key=v["demo_cache_key"],
        n_train_trajs=v["n_train_trajs"],
        n_test_trajs=10 if MODE == "local_docker" else 50,
        n_passes_per_epoch=1,
        horizon=v["horizon"],
        n_epochs=v["n_epochs"],
        learning_rate=1e-2,
        no_improvement_tolerance=10,
        shuffler=ConoptParticleEnv.shuffler() if v["use_shuffler"] else None,
        batch_size=16,
    )

    algo.train()


for v in variants:

    conopt_run_experiment(
        run_task,
        use_cloudpickle=True,
        exp_prefix="conopt-particle-4-2",
        mode=MODE,
        n_parallel=0,
        seed=v["seed"],
        snapshot_mode="last",
        variant=v,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        use_gpu=USE_GPU,
    )
    # sys.exit()
