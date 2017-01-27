from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.new_analogy.policies.residual_policy import ResidualPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import pickle

"""
PI^2 with the reward shaped as distance to a predefined pose
"""

MODE = "local_docker"
N_PARALLEL = 1


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11]


def run_task(vv):
    from gpr_package.bin import tower_fetch_policy as tower
    import numpy as np
    task_id = tower.get_task_from_text("ab")
    num_substeps = 1
    horizon = 1000 // num_substeps
    expr = tower.SimFetch(nboxes=2, horizon=horizon, mocap=True, obs_type="full_state", num_substeps=num_substeps)
    policy = tower.FetchPolicy(task_id)
    env = expr.make(task_id=task_id)
    env.seed(vv["seed"])
    #
    np.random.seed(vv["seed"])

    xinit = env.world.sample_xinit()
    ob = env.reset_to(xinit)
    acts = []
    xs = []
    total_reward = 0
    for _ in range(horizon):
        action = policy.get_action(ob)
        acts.append(action)
        xs.append(env.x)
        ob, reward, done, _ = env.step(action)
        total_reward += reward

    acts = np.asarray(acts)

    start_t = 0
    end_t = 100
    xinit = xs[start_t]
    final_x = xs[end_t]
    # acts = np.asarray(acts)
    # xs = np.asarray(xs)
    # acts_diff = np.sum(np.square(acts[1:] - acts[:-1]), axis=-1)
    # xs_diff = np.sum(np.square(xs[1:] - xs[:-1]), axis=-1)
    # print("Final reward: ", reward)
    # print("Total reward: ", total_reward)

    from sandbox.rocky.new_analogy.algos.pi2 import PI2
    import numpy as np
    import gpr.env
    logger.log("Launching PI2...")
    from sandbox.rocky.new_analogy.gpr_ext.rewards import SenseDistReward

    horizon = end_t - start_t#100#0

    means = np.array(
        [0.26226245, 0.04312864, 0.61919527, 0., 1.57,
         0., 0.13145038, 0.13145038]
    )
    stds = np.array([9.87383461e-02, 1.34582481e-01, 5.18472679e-02,
                     0.00000000e+00, 4.67359484e-12, 0.00000000e+00,
                     9.91322751e-01, 9.91322751e-01])

    # from sandbox.rocky.new_analogy.gpr_ext.rewards import TimeReward
    # from gpr.reward import PenaltyReward
    reward = -SenseDistReward("qpos", final_x[:24], metric="GPS:1,1e-2,1e-4") #- 1e-5*PenaltyReward("qacc")#\
             #- SenseDistReward("qvel", final_x[24:], metric="L2")

    expr = tower.SimFetch(nboxes=2, horizon=horizon, mocap=False, obs_type="full_state", num_substeps=num_substeps)
    policy = tower.FetchPolicy(task_id)
    env = expr.make(task_id=task_id)

    algo = PI2(
        env=gpr.env.Env(
            world_builder=env.world_builder,
            reward=reward,#,log:1e-4"),
            horizon=horizon,
            task_id=env.task_id,
            delta_reward=env.delta_reward,
            delta_obs=env.delta_obs
        ),
        time_multiplier=np.arange(horizon),
        xinit=xinit,
        num_iterations=15,
        particles=500,
        init_cov=1.,#stds**2 + 1e-6,
        #init_k=acts[start_t:end_t],#[100:300],#np.tile(means.reshape((1, -1)), (horizon, 1)),#np.(acts),
    )
    algo.train()


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:

    kwargs = dict(
        use_cloudpickle=True,
        exp_prefix="tower-pi2",
        exp_name="tower-pi2",
        mode=MODE,
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=N_PARALLEL,
        env=dict(CUDA_VISIBLE_DEVICES="4", PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
        variant=v,
        seed=v["seed"],
    )

    if MODE == "local":
        del kwargs["env"]["PYTHONPATH"]  # =
    else:
        kwargs = dict(
            kwargs,
            docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170114",
            docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        )

    run_experiment_lite(
        run_task,
        **kwargs,
    )
