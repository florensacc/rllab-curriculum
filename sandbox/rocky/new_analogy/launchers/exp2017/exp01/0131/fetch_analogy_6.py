from collections import OrderedDict

from torch import optim

from rllab.misc import logger
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.new_analogy.exp_utils import run_local
from sandbox.rocky.th.core.modules import DilatedConvNetSpec

"""
Dilated convolution for conditioning on demonstrations
"""

DEBUG = True  # False#True#False

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
        return [16]

    @variant
    def per_demo_batch_size(self):
        return [500]

    @variant
    def per_demo_full_traj(self):
        return [True]

    @variant
    def n_train_tasks(self):
        # return [2]
        return [2]  # 8]

    @variant
    def n_test_tasks(self):
        return [0]

    @variant
    def n_updates_per_epoch(self):
        return [100]
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
        return [False]  # True, False]


def run_task(vv):
    from sandbox.rocky.new_analogy.th.algos.fetch_analogy_trainer import FetchAnalogyTrainer
    from sandbox.rocky.new_analogy import fetch_utils
    from sandbox.rocky.new_analogy.th.policies import ConvAnalogyPolicy
    from sandbox.rocky.th import tensor_utils

    template_env = fetch_utils.fetch_env(
        horizon=2000,
        height=5,
    )
    env_spec = fetch_utils.discretized_env_spec(
        template_env.spec,
        disc_intervals=fetch_utils.disc_intervals
    )
    policy = ConvAnalogyPolicy(
        env_spec=env_spec,
        encoder_spec=DilatedConvNetSpec(
            dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
            n_channels=64,
            kernel_size=2,
            causal=False,
            weight_norm=True,
        ),
        obs_embedding_hidden_sizes=(128,),
        decoder_spec=DilatedConvNetSpec(
            dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
            n_channels=64,
            kernel_size=2,
            causal=True,
            weight_norm=True,
        ),
        weight_norm=True,
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
        per_demo_full_traj=vv["per_demo_full_traj"],
        n_updates_per_epoch=vv["n_updates_per_epoch"],
        n_train_tasks=vv["n_train_tasks"],
        n_test_tasks=vv["n_test_tasks"],
        n_configurations=100 if DEBUG else 1000,
        n_eval_paths_per_task=10,
        evaluate_policy=False,
    ).train()


variants = VG().variants()
print("#Experiments:", len(variants))

for v in variants:
    # run_cirrascale(
    # run_task(v)
    run_local(  # )
        run_task,
        exp_name="fetch-analogy-6",
        variant=v,
        seed=v["seed"],
        n_parallel=0,
    )
    exit()
