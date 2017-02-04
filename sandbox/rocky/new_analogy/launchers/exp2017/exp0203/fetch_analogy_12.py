from collections import OrderedDict

from torch import optim

from rllab.misc import logger
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.new_analogy.exp_utils import run_local, run_local_docker, run_cirrascale
from sandbox.rocky.s3 import resource_manager
from sandbox.rocky.th import ops

"""
Testing ground truth again, this time also on test tasks
"""

DEBUG = False#True#False  # True  # False#True#False

OPTIMIZERS = OrderedDict([
    # ("adamax_1e-2", lambda policy: optim.Adamax(policy.parameters(), lr=1e-2)),
    # ("adamax_1e-3", ops.get_optimizer('adamax', lr=0)),#1e-3)),
    ("adamax_1e-3", ops.get_optimizer('adamax', lr=1e-3)),
    # ("adam_1e-2", lambda policy: optim.Adam(policy.parameters(), lr=1e-2)),
    # ("adam_1e-3", lambda policy: optim.Adam(policy.parameters(), lr=1e-3)),
    # ("rmsprop_1e-2", lambda policy: optim.RMSprop(policy.parameters(), lr=1e-2)),
    # ("rmsprop_1e-3", lambda policy: optim.RMSprop(policy.parameters(), lr=1e-3)),
])


class VG(VariantGenerator):
    @variant
    def demo_batch_size(self):
        return [512]

    @variant
    def per_demo_batch_size(self):
        return [1]

    @variant
    def per_demo_full_traj(self):
        return [False]

    @variant
    def n_train_tasks(self):
        if DEBUG:
            return [2]
        return [16]

    @variant
    def n_test_tasks(self, n_train_tasks):
        if DEBUG:
            return [0]
        return [4]  # 0, 4]
            # return [0]

    @variant
    def n_updates_per_epoch(self):
        if DEBUG:
            return [1000]
        return [10000]

    @variant
    def seed(self):
        return [100]

    @variant
    def optimizer_type(self):
        return list(OPTIMIZERS.keys())


def run_task(vv):
    from sandbox.rocky.new_analogy.th.algos.fetch_analogy_trainer import FetchAnalogyTrainer
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.new_analogy.launchers.exp2017.exp0203.ground_truth_analogy_policy import GroundTruthAnalogyPolicy

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
        embedding_dim=256,
        hidden_sizes=(256, 256),
    )

    optimizer = OPTIMIZERS[vv["optimizer_type"]]

    FetchAnalogyTrainer(
        template_env=template_env,
        policy=policy,
        optimizer=optimizer,
        demo_batch_size=vv["demo_batch_size"],
        per_demo_batch_size=vv["per_demo_batch_size"],
        per_demo_full_traj=vv["per_demo_full_traj"],
        n_updates_per_epoch=vv["n_updates_per_epoch"],
        n_train_tasks=vv["n_train_tasks"],
        n_test_tasks=vv["n_test_tasks"],
        n_configurations=100 if DEBUG else 1000,
        n_eval_paths_per_task=100,
        evaluate_policy=True,
    ).train()


variants = VG().variants()
print("#Experiments:", len(variants))

for v in variants:
    run_cirrascale(
    # run_local(
        run_task,
        exp_name="fetch-analogy-12",
        variant=v,
        seed=v["seed"],
        n_parallel=0,
    )
    # if DEBUG:
    #     exit()
