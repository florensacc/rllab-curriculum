from collections import OrderedDict

from torch import optim

from rllab.misc import logger
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.new_analogy.exp_utils import run_cirrascale

"""
Feed in final state with deeper networks
"""

DEBUG = False#True#False

OPTIMIZERS = OrderedDict([
    # ("adamax_1e-2", lambda policy: optim.Adamax(policy.parameters(), lr=1e-2)),
    ("adamax_1e-3", lambda policy: optim.Adamax(policy.parameters(), lr=1e-3)),
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
    def n_train_tasks(self):
        # return [2]
        return [16]

    @variant
    def n_test_tasks(self):
        return [4]

    @variant
    def n_updates_per_epoch(self):
        # return [100]
        if DEBUG:
            return [1000]
        else:
            return [10000]

    @variant
    def seed(self):
        return [100]

    @variant
    def optimizer_type(self):
        return list(OPTIMIZERS.keys())

    @variant
    def batch_norm(self):
        return [True, False]


def run_task(vv):
    from sandbox.rocky.new_analogy.th.algos.fetch_analogy_trainer import FetchAnalogyTrainer
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.new_analogy.th.policies import FinalStateAnalogyPolicy
    from sandbox.rocky.th import tensor_utils

    template_env = fetch_utils.fetch_env(
        horizon=2000,
        height=5,
    )
    env_spec = fetch_utils.discretized_env_spec(
        template_env.spec,
        disc_intervals=fetch_utils.disc_intervals
    )
    policy = FinalStateAnalogyPolicy(
        env_spec=env_spec,
        hidden_sizes=(512, 512),
        embedding_hidden_sizes=(512, 512),
        batch_norm=vv["batch_norm"],
    )

    if tensor_utils.is_cuda():
        logger.log("Uploading policy to cuda")
        policy.cuda()

    optimizer = OPTIMIZERS[vv["optimizer_type"]](policy)

    FetchAnalogyTrainer(
        template_env=template_env,
        policy=policy,
        optimizer=optimizer,
        demo_batch_size=vv["demo_batch_size"],
        per_demo_batch_size=vv["per_demo_batch_size"],
        n_updates_per_epoch=vv["n_updates_per_epoch"],
        n_train_tasks=vv["n_train_tasks"],
        n_test_tasks=vv["n_test_tasks"],
        n_configurations=1000,
        n_eval_paths_per_task=10,
    ).train()


variants = VG().variants()
print("#Experiments:", len(variants))

for v in variants:
    run_cirrascale(
    # run_local(#)
        run_task,
        exp_name="fetch-analogy-5-5",
        variant=v,
        seed=v["seed"],
        n_parallel=0,
    )
    # exit()
