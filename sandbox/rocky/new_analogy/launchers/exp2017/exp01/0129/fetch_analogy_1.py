from sandbox.rocky.new_analogy.exp_utils import run_ec2

"""
Try without dagger, new obs space
"""

DEBUG = False#True#False  # True#False#True


def run_task(vv):
    from sandbox.rocky.new_analogy.th.algos.fetch_analogy_trainer import FetchAnalogyTrainer

    if DEBUG:
        FetchAnalogyTrainer(
            batch_size=vv["batch_size"],
            rnn_hidden_size=vv["hidden_size"],
            n_updates_per_epoch=10,
            train_ratio=0.2,
        ).train()
    else:
        FetchAnalogyTrainer(
            batch_size=vv["batch_size"],
            rnn_hidden_size=vv["hidden_size"],
            n_updates_per_epoch=vv["n_updates_per_epoch"],
            train_ratio=0.8,
            n_configurations=1000,
            n_eval_paths_per_task=10,
        ).train()


for batch_size in [16, 64, 256]:  # , 1024, 4096]:

    for hidden_size in [32, 64, 128]:

        for n_updates_per_epoch in [1000, 10000]:

            for seed in [100]:

                # run_cirrascale(
                run_ec2(#)
                    run_task,
                    exp_name="fetch-analogy-1-3",
                    variant=dict(
                        batch_size=batch_size,
                        hidden_size=hidden_size,
                        n_updates_per_epoch=n_updates_per_epoch,
                        seed=seed,
                    ),
                    seed=seed,
                    n_parallel=0,
                    disk_size=100,
                    # use_gpu=False,
                )
