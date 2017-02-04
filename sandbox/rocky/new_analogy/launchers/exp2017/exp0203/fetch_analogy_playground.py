from multiprocessing import cpu_count

from sandbox.rocky.new_analogy.exp_utils import run_local
from sandbox.rocky.th import ops
import numpy as np

"""
Testing ground truth again, this time also on test tasks
"""


def run_task(_):
    from sandbox.rocky.new_analogy.th.algos.fetch_analogy_trainer import FetchAnalogyTrainer
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.new_analogy.launchers.exp2017.exp0203.ground_truth_analogy_policy import GroundTruthAnalogyPolicy
    from sandbox.rocky.new_analogy.th.algos.fetch_analogy_trainer import AnalogyDataset

    task_id = "ab"

    horizon = 500
    n_paths = 100
    height = 2

    env = fetch_utils.fetch_env(horizon=horizon, height=height, task_id=fetch_utils.get_task_from_text(task_id))
    env_spec = fetch_utils.discretized_env_spec(
        env.spec,
        disc_intervals=fetch_utils.disc_intervals
    )

    policy = GroundTruthAnalogyPolicy(
        env_spec=env_spec,
        hidden_sizes=(64, 64),
        # embedding doesn't matter here
        embedding_dim=1,
    )

    demo_policy = fetch_utils.fetch_discretized_prescribed_policy(env=env, disc_intervals=fetch_utils.disc_intervals)
    print("Generating xinits")
    xinits = fetch_utils.collect_xinits(height=height, seeds=np.arange(n_paths))
    print("Generating paths")
    paths = fetch_utils.new_policy_paths(
        seeds=np.arange(n_paths),
        env=env,
        policy=demo_policy,
        noise_levels=[0., 1e-3, 1e-2],
        xinits=xinits,
    )

    print("Converting to discrete actions")
    # map to discrete-valued actions
    for p in paths:
        disc_actions = []
        for disc_idx in range(3):
            cur_actions = p["actions"][:, disc_idx]
            bins = np.asarray(fetch_utils.disc_intervals[disc_idx])
            disc_actions.append(np.argmin(np.abs(cur_actions[:, None] - bins[None, :]), axis=1))
        disc_actions.append(np.cast['uint8'](p["actions"][:, -1] == 1))
        flat_actions = env_spec.action_space.flatten_n(np.asarray(disc_actions).T)
        p["actions"] = flat_actions
    print("Converted")

    is_success = lambda p: env.wrapped_env._is_success(p) and len(p["rewards"]) == horizon
    # filter out failed paths
    print("Success rate for discretized prescribed policy", np.mean(list(map(is_success, paths))))
    paths = np.asarray(list(filter(is_success, paths)))

    train_dataset = AnalogyDataset()
    # empty test dataset; only testing on trained tasks
    test_dataset = AnalogyDataset()
    train_dataset.add_paths(task_id=task_id, paths=paths, env=env)

    # this class is intended to train analogies, but here we're just training on one task which is much faster,
    # but there will be some boilerplate for using this class
    FetchAnalogyTrainer(
        template_env=env,
        policy=policy,
        optimizer=ops.get_optimizer('adamax', lr=1e-3),
        demo_batch_size=512,
        per_demo_batch_size=1,
        per_demo_full_traj=False,
        n_updates_per_epoch=1000,
        n_train_tasks=1,
        n_test_tasks=0,
        n_configurations=n_paths,
        n_eval_paths_per_task=10,
        evaluate_policy=True,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        xinits=xinits,
    ).train()


run_local(
    run_task,
    exp_name="fetch-analogy-playground",
    seed=0,
    # This is only used to generate xinits and demo paths
    n_parallel=cpu_count(),
)
