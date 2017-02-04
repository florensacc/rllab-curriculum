from sandbox.rocky.new_analogy.exp_utils import run_cirrascale

"""
Try without dagger, new obs space
"""

DEBUG = False#True#False#True#False  # True#False#True


def run_task(vv):
    from sandbox.rocky.new_analogy.th.algos.fetch_analogy_trainer import FetchAnalogyTrainer
    from sandbox.rocky.new_analogy.th.policies import LSTMAnalogyPolicy
    from sandbox.rocky.new_analogy import fetch_utils

    # if is_cirrascale():
    #     resource_manager.local_resource_path = "/shared-data/resource"

    template_env = fetch_utils.fetch_env(
        horizon=2000,
        height=5,
    )
    env_spec = fetch_utils.discretized_env_spec(
        template_env.spec,
        disc_intervals=fetch_utils.disc_intervals
    )
    policy = LSTMAnalogyPolicy(
        env_spec=env_spec,
        rnn_hidden_size=vv["rnn_hidden_size"],
        hidden_sizes=vv["hidden_sizes"],
    )

    if DEBUG:
        FetchAnalogyTrainer(
            demo_batch_size=16,
            per_demo_batch_size=256,
            policy=policy,
            n_updates_per_epoch=1000,
            n_train_tasks=1,
            n_test_tasks=0,
            n_configurations=1000,
            evaluate_policy=True,
        ).train()
    else:
        FetchAnalogyTrainer(
            policy=policy,
            demo_batch_size=vv["demo_batch_size"],
            per_demo_batch_size=vv["demo_batch_size"],
            n_updates_per_epoch=vv["n_updates_per_epoch"],
            n_train_tasks=16,
            n_test_tasks=4,
            n_configurations=1000,
            n_eval_paths_per_task=10,
        ).train()


for demo_batch_size in [16, 32, 64]:

    per_demo_batch_size = 256

    for rnn_hidden_size in [32, 64, 128]:

        for hidden_sizes in [(256, 256)]:

            for n_updates_per_epoch in [10000]:

                for seed in [100]:
                    run_cirrascale(
                    # run_local_docker(
                        run_task,
                        exp_name="fetch-analogy-2-1",
                        variant=dict(
                            demo_batch_size=demo_batch_size,
                            per_demo_batch_size=per_demo_batch_size,
                            rnn_hidden_size=rnn_hidden_size,
                            hidden_sizes=hidden_sizes,
                            n_updates_per_epoch=n_updates_per_epoch,
                            seed=seed,
                        ),
                        seed=seed,
                        n_parallel=0,
                    )
                    # exit()
