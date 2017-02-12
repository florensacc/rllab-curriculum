from sandbox.rocky.new_analogy.exp_utils import run_cirrascale

"""
Feed in ground truth task id
"""

DEBUG = False#True#False#True


def run_task(vv):
    from sandbox.rocky.new_analogy.th.algos.fetch_analogy_trainer import FetchAnalogyTrainer
    from sandbox.rocky.new_analogy import fetch_utils
    # from sandbox.rocky.new_analogy.policies.final_state_analogy_policy import FinalStateAnalogyPolicy
    from sandbox.rocky.new_analogy.th.policies.ground_truth_analogy_policy import GroundTruthAnalogyPolicy

    template_env = fetch_utils.fetch_env(
        horizon=2000,
        height=5,
    )
    env_spec = fetch_utils.discretized_env_spec(
        template_env.spec,
        disc_intervals=fetch_utils.disc_intervals
    )
    policy = GroundTruthAnalogyPolicy(
        env_spec=env_spec,
        hidden_sizes=vv["hidden_sizes"],
    )

    if DEBUG:
        FetchAnalogyTrainer(
            template_env=template_env,
            policy=policy,
            demo_batch_size=vv["demo_batch_size"],
            per_demo_batch_size=vv["per_demo_batch_size"],
            n_updates_per_epoch=100,
            n_train_tasks=1,
            n_test_tasks=0,
            n_configurations=100,
            evaluate_policy=True,
        ).train()
    else:
        FetchAnalogyTrainer(
            template_env=template_env,
            policy=policy,
            demo_batch_size=vv["demo_batch_size"],
            per_demo_batch_size=vv["per_demo_batch_size"],
            n_updates_per_epoch=vv["n_updates_per_epoch"],
            n_train_tasks=vv["n_train_tasks"],
            n_test_tasks=vv["n_test_tasks"],
            n_configurations=1000,
            n_eval_paths_per_task=10,
        ).train()


for demo_batch_size in [512]:#, 1024, 2048, 4096]:

    per_demo_batch_size = 1

    for n_train_tasks in [1, 2, 4, 8, 16]:

        n_test_tasks = 0#1

        for hidden_sizes in [(256, 256), (512, 512, 512)]:

            for n_updates_per_epoch in [10000]:

                for seed in [100]:
                    run_cirrascale(
                    # run_local(
                        run_task,
                        exp_name="fetch-analogy-4-2",
                        variant=dict(
                            demo_batch_size=demo_batch_size,
                            per_demo_batch_size=per_demo_batch_size,
                            hidden_sizes=hidden_sizes,
                            n_updates_per_epoch=n_updates_per_epoch,
                            n_train_tasks=n_train_tasks,
                            n_test_tasks=n_test_tasks,
                            seed=seed,
                        ),
                        seed=seed,
                        n_parallel=0,
                        # profile=True,
                        # use_gpu=True,
                    )
                    # exit()
