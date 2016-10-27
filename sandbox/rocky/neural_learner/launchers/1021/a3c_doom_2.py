from rllab import config
from rllab.misc.instrument import run_experiment_lite, VariantGenerator, variant

import logging

USE_GPU = False  # True
MODE = "ec2"

if MODE == "local_docker":
    if USE_GPU:
        env = dict(CUDA_VISIBLE_DEVICES="3")
    else:
        env = dict(CUDA_VISIBLE_DEVICES="")
else:
    env = dict()

env['MKL_NUM_THREADS'] = '1'
env['NUMEXPR_NUM_THREADS'] = '1'
env['OMP_NUM_THREADS'] = '1'

config.DOCKER_IMAGE = "dementrock/rllab3-vizdoom-gpu-cuda80:cig"


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def entropy_bonus(self):
        return [0.01]

    @variant
    def batch_size(self):
        yield 50000

    @variant
    def n_itr(self):
        yield 500


vg = VG()
variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:
    def run_task(v):
        from sandbox.rocky.neural_learner.async_rl.agents.a3c_agent import A3CAgent
        from sandbox.rocky.neural_learner.async_rl.algos.a3c_ale import A3CALE
        from sandbox.rocky.neural_learner.envs.doom_two_goal_env import DoomTwoGoalEnv
        from sandbox.rocky.neural_learner.envs.doom_goal_finding_maze_env import DoomGoalFindingMazeEnv

        env = DoomTwoGoalEnv(
            rescale_obs=(30, 40),
            reset_map=True,
            living_reward=-0.01,
        )

        agent = A3CAgent(
            n_actions=env.action_dim,
            beta=v["entropy_bonus"],
            sync_t_gap_limit=1000,
            clip_reward=False,
        )

        if MODE == "local_docker":
            n_processes = 32  # 1#3
        else:
            n_processes = 36

        algo = A3CALE(
            n_processes=n_processes,
            env=env,
            agent=agent,
            logging_level=logging.INFO,
            eval_frequency=v["batch_size"],
            total_steps=v["batch_size"] * v["n_itr"],
            eval_n_runs=10,
            horizon=300,
            eval_horizon=300,
            seeds=None,
        )

        algo.train()


    config.AWS_INSTANCE_TYPE = "c4.8xlarge"
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.675'
    config.AWS_REGION_NAME = 'us-west-1'
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    run_experiment_lite(
        run_task,
        exp_prefix="a3c-doom-2",
        mode=MODE,
        use_gpu=USE_GPU,
        use_cloudpickle=True,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        snapshot_mode="last",
        env=env,
        seed=vv["seed"],
        variant=vv,
    )
    # break
