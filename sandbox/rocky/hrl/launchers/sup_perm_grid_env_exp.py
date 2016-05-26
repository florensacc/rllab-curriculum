from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from sandbox.rocky.hrl.algos.supervised_markov_polopt import SupervisedMarkovPolopt
from sandbox.rocky.hrl.algos.supervised_recurrent_polopt import SupervisedRecurrentPolopt
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())
from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("bonus_coeff", [0., 0.1, 0.01, 1.0, 10.0, 100.0])
vg.add("learning_rate", [1e-3, 5e-3, 1e-4, 5e-4])

variants = vg.variants()

print("#Variants: %d" % len(variants))

for v in variants:

    env = PermGridEnv(
        size=5,
        n_objects=5,
        object_seed=0,
        random_restart=False,
        training_paths_ratio=0.5
    )

    policy = StochasticGRUPolicy(
        env_spec=env.spec,
        n_subgoals=5,
        use_decision_nodes=False,
        random_reset=False
    )
    # policy = CategoricalMLPPolicy(env_spec=env.spec)
    algo = SupervisedRecurrentPolopt(
        env,
        policy,
        n_on_policy_samples=10000,
        max_path_length=100,
        bonus_coeff=v["bonus_coeff"],
        learning_rate=v["learning_rate"],
    )
    # algo = SupervisedMarkovPolopt(env, policy, n_test_samples=1000)

    run_experiment_lite(
        algo.train(),
        n_parallel=2,
        seed=v["seed"],
        mode="lab_kube",
        exp_prefix="info_reg_sup",
    )
