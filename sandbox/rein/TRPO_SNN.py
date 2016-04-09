import os
from rllab.mdp.box2d.cartpole_mdp import CartpoleMDP
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.policy.mean_std_nn_policy import MeanStdNNPolicy
from rllab.baseline.nn_baseline import NNBaseline
from rllab.algo.trpo_snn import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(1)
etas = [0.1, 0.5, 1.0, 5.0]
replay_pools = [True, False]
kl_ratios = [True]
reverse_kl_regs = [True]
param_cart_product = itertools.product(
    reverse_kl_regs, kl_ratios, replay_pools, etas, seeds
)

for reverse_kl_reg, kl_ratio, replay_pool, eta, seed in param_cart_product:
    
    mdp = CartpoleMDP()

    policy = MeanStdNNPolicy(
        mdp=mdp,
        hidden_sizes=(32,),
    )

    baseline = NNBaseline(
        mdp=mdp,
        hidden_sizes=(32,),
    )

    algo = TRPO(
        batch_size=1000,
        whole_paths=False,
        max_path_length=100,
        n_itr=100,
        step_size=0.01,
        eta=eta,
        eta_discount=0.99,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_reverse_kl_reg=reverse_kl_reg,
        use_replay_pool=replay_pool,
        use_kl_ratio=kl_ratio,
    )

    run_experiment_lite(
        algo.train(mdp=mdp, policy=policy, baseline=baseline),
        exp_prefix="cartpole",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="openai_kube",
        dry=False,
    )
