import os

from sandbox.rocky.analogy.utils import conopt_run_experiment

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale

from rllab.misc.instrument import VariantGenerator, variant, run_experiment_lite

"""
Test copy task on more architectures
"""

USE_GPU = True  # True
USE_CIRRASCALE = True  # True
TYPE_FILTER = "maxwell"
# MODE = "local_docker"
MODE = launch_cirrascale(type_filter=TYPE_FILTER)
ENV = "copy"
VERSION = "v3"


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def n_train_trajs(self):
        return [10000]#, 50000, 100000]

    @variant
    def use_shuffler(self):
        return [True]

    @variant
    def state_include_action(self):
        return [False]

    @variant
    def horizon(self):
        return [100]

    @variant
    def n_epochs(self):
        return [1000]

    @variant
    def seq_length(self):
        return [20, 40, 80]

    @variant
    def n_elems(self):
        yield 5

    @variant
    def obs_size(self):
        yield (30, 30)

    @variant
    def batch_size(self):
        return [128]

    @variant
    def network(self):
        return ["attention_multiplicative", "attention_monotonic", "attention", "double_rnn"]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))


def run_task(v):
    from sandbox.rocky.analogy.algos.trainer import Trainer
    from sandbox.rocky.analogy.envs.copy_env import CopyEnv, CopyPolicy
    from sandbox.rocky.analogy.demo_collector.policy_demo_collector import PolicyDemoCollector
    from sandbox.rocky.analogy.policies.modular_analogy_policy import ModularAnalogyPolicy
    from sandbox.rocky.analogy.networks.copy import double_rnn, attention
    from sandbox.rocky.analogy.rnn_cells import AttentionCell, AttentionCell_Monotonic, AttentionCell_Multiplicative
    from sandbox.rocky.tf.envs.base import TfEnv

    env_args = dict(
        min_seq_length=1,#v["seq_length"],
        max_seq_length=v["seq_length"],
        n_elems=v["n_elems"],
    )

    env = TfEnv(CopyEnv(
        seed=0,
        **env_args
    ))

    if v["network"] == "double_rnn":
        net = double_rnn.Net()
    elif v["network"] == "attention":
        net = attention.Net(attention_cell_cls=AttentionCell)
    elif v["network"] == "attention_monotonic":
        net = attention.Net(attention_cell_cls=AttentionCell_Monotonic)
    elif v["network"] == "attention_multiplicative":
        net = attention.Net(attention_cell_cls=AttentionCell_Multiplicative)
    else:
        raise NotImplementedError

    policy = ModularAnalogyPolicy(
        env_spec=env.spec,
        name="policy",
        net=net,
    )

    def demo_cache_key(trainer, env, policy):
        return "-".join([
            ENV,
            VERSION,
            "minseq" + str(env.min_seq_length),
            "maxseq" + str(env.max_seq_length),
            "n" + str(trainer.n_train_trajs)
        ])

    generalization_env_cls_list = []
    for seq_len in range(v["seq_length"]+20, 101, 20):
        generalization_env_cls_list.append((str(seq_len), TfEnv.wrap(CopyEnv, **dict(env_args, min_seq_length=seq_len,
                                                                              max_seq_length=seq_len))))

    algo = Trainer(
        policy=policy,
        normalize=False,
        env_cls=TfEnv.wrap(
            CopyEnv,
            **env_args
        ),
        generalization_env_cls_list=generalization_env_cls_list,
        demo_collector=PolicyDemoCollector(
            policy_cls=CopyPolicy
        ),
        skip_eval=False,
        demo_cache_key=demo_cache_key,
        n_train_trajs=v["n_train_trajs"],
        n_test_trajs=10 if MODE == "local_docker" else 50,
        n_passes_per_epoch=1,
        horizon=v["horizon"],
        n_epochs=v["n_epochs"],
        learning_rate=1e-2,
        no_improvement_tolerance=10,
        batch_size=v["batch_size"],
        use_curriculum=False,
    )

    algo.train()


for v in variants:
    conopt_run_experiment(
        run_task,
        use_cloudpickle=True,
        exp_prefix="analogy-copy-2-1",
        mode=MODE,
        n_parallel=8,
        seed=v["seed"],
        snapshot_mode="last",
        variant=v,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        use_gpu=USE_GPU,
        type_filter=TYPE_FILTER,
    )
    # sys.exit()
