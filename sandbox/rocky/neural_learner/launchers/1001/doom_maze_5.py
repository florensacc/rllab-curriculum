from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant


USE_GPU = False#True#False  # True#False
USE_CIRRASCALE = True
MODE = "lab_kube"

# config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu-cuda80"
config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-tester"
config.KUBE_DEFAULT_NODE_SELECTOR = {
    "aws/type": "c4.2xlarge",
}
config.KUBE_DEFAULT_RESOURCES = {
    "requests": {
        "cpu": 8 * 0.75,
        "memory": "10Gi",
    },
}
env = dict(CUDA_VISIBLE_DEVICES="")


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [10*x+1 for x in range(1, 4)]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:  # [:10]:

    def run_task(v):
        from rllab.policies.uniform_control_policy import UniformControlPolicy
        from sandbox.rocky.tf.envs.base import TfEnv
        from sandbox.rocky.tf.algos.nop import NOP
        from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv
        from sandbox.rocky.neural_learner.envs.doom_default_wad_env import DoomDefaultWadEnv
        from rllab.baselines.zero_baseline import ZeroBaseline
        from rllab import config
        import os

        env = TfEnv(
            DoomDefaultWadEnv(
                os.path.join(config.PROJECT_PATH, "sandbox/rocky/neural_learner/envs/wads/vizdoom_levels/basic.wad"),
                vectorized=True,
                verbose_debug=True
            )
        )
        # env = TfEnv(DoomGoalFindingMazeEnv())
        policy = UniformControlPolicy(env.spec)
        baseline = ZeroBaseline(env.spec)

        algo = NOP(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=1000,
            max_path_length=10,
            sampler_args=dict(n_envs=32),
        )
        algo.train()


    run_experiment_lite(
        run_task,
        exp_prefix="doom_maze_5_9",
        mode=MODE,
        n_parallel=0,
        seed=v["seed"],
        use_gpu=USE_GPU,
        use_cloudpickle=True,
        # variant=v,
        snapshot_mode="last",
        env=env,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        sync_log_on_termination=False,
        # docker_args=" -m 10g --cpuset-cpus='0,1,2,3'  "
    )
    # sys.exit()
