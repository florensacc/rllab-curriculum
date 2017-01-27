from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.new_analogy.policies.residual_policy import ResidualPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import pickle

"""
Behavior clone single trajectory
"""

MODE = "local_docker"  # _docker"  # _docker"
# MODE = launch_cirrascale("pascal")
N_PARALLEL = 1  # 8


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]  # , 41, 51]


def run_task(vv):
    from gpr_package.bin import tower_fetch_policy as tower
    import numpy as np
    # from gpr_package.bin import tower_copter_policy as tower
    task_id = tower.get_task_from_text("ab")
    num_substeps = 1#5
    horizon = 1000 // num_substeps
    ctrl_smoothness_pen = 1
    # expr = tower.Experiment(nboxes=2, horizon=horizon, num_substeps=5)
    # policy = tower.CopterPolicy(task_id)
    #
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
    xs = np.asarray(xs)
    acts_diff = np.sum(np.square(acts[1:] - acts[:-1]), axis=-1)
    xs_diff = np.sum(np.square(xs[1:] - xs[:-1]), axis=-1)
    ctrl_pen = ctrl_smoothness_pen * np.sum(np.clip(acts_diff / (xs_diff + 1e-8), -10, 10))
    print("Final reward: ", reward)
    print("Total reward: ", total_reward)
    print("Total Ctrl Pen: ", ctrl_pen)

    from sandbox.rocky.new_analogy.algos.pi2 import PI2
    import numpy as np
    logger.log("Launching PI2...")
    algo = PI2(
        env=env,
        xinit=xinit,
        num_iterations=15,
        particles=500,
        init_cov=0.1,
        init_k=np.asarray(acts),
        ctrl_smoothness_pen=ctrl_smoothness_pen,#0.001,
    )
    algo.train()
    # from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    # import tensorflow as tf
    # from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    # from sandbox.rocky.new_analogy.algos.ff_bc_trainer import Trainer
    # from sandbox.rocky.s3.resource_manager import resource_manager
    #
    # with tf.Session() as sess:
    #     logger.log("Loading data...")
    #     file_name = resource_manager.get_file("tower_copter_paths_ab_crippled_100")
    #     with open(file_name, 'rb') as f:
    #         paths = pickle.load(f)
    #     logger.log("Loaded")
    #
    #     xinits = []
    #     for path in paths:
    #         xinits.append(path["env_infos"]["x"][0])
    #
    #     task_id = tower.get_task_from_text("ab")
    #
    #     env = TfEnv(
    #         GprEnv(
    #             "tower",
    #             task_id=task_id,
    #             experiment_args=dict(nboxes=2, horizon=1000),
    #             xinits=xinits[:1],
    #         )
    #     )
    #
    #     policy = ResidualPolicy(env_spec=env.spec, wrapped_policy=GaussianMLPPolicy(
    #         env_spec=env.spec,
    #         hidden_sizes=(256, 256, 256, 256),#128, 128),
    #         hidden_nonlinearity=tf.nn.tanh,
    #         name="policy"
    #     ))
    #
    #     algo = Trainer(
    #         env=env,
    #         policy=policy,
    #         paths=paths[:1],
    #         n_epochs=500,
    #         evaluate_performance=True,
    #         n_passes_per_epoch=1000,
    #         train_ratio=1.,#0.9,
    #         max_path_length=1000,
    #         n_eval_trajs=1,
    #         eval_batch_size=1000,
    #         n_eval_envs=1,
    #         threshold=4.,
    #         batch_size=128,
    #         n_slices=10,
    #     )
    #
    #     algo.train(sess=sess)


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
        # pre_commands=["pip install --upgrade numpy"]
    )
