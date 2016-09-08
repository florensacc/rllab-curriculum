


from rllab.misc import instrument
from sandbox.rocky.hrl.envs.seaquest_grid_world_env import SeaquestGridWorldEnv

from rllab.algos.trpo import TRPO
from rllab.spaces.discrete import Discrete
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.core.network import ConvNetwork
from sandbox.rocky.hrl.batch_hrl import BatchHRL
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy
from sandbox.rocky.hrl.subgoal_baseline import SubgoalBaseline
from sandbox.rocky.hrl.mi_evaluator.state_based_mi_evaluator import StateBasedMIEvaluator
from sandbox.rocky.hrl.mi_evaluator.state_based_value_mi_evaluator import StateBasedValueMIEvaluator
from sandbox.rocky.hrl.core.network import ConvMergeNetwork
from sandbox.rocky.hrl.regressors.shared_network_auto_mlp_regressor import SharedNetworkAutoMLPRegressor
import lasagne.nonlinearities as NL
import sys
import numpy as np

HIERARCHICAL = True

instrument.stub(globals())

if HIERARCHICAL:

    batch_size = 4000

    grid_size = 8

    env = SeaquestGridWorldEnv(
        size=grid_size,
        n_bombs=grid_size / 2,
        guided_observation=False,
    )

    guided_obs_size = grid_size + grid_size + 2


    from rllab import config

    config.AWS_INSTANCE_TYPE = "c4.2xlarge"

    vg = instrument.VariantGenerator()

    vg.add("seed", [11, 21, 31])#, 21, 31, 41, 51])
    vg.add("n_subgoals", [5, 10, 15])#, 10, 15, 20, 25, 30])
    vg.add("mi_coeff", [0, 0.01, 0.1, 1])#0.01])#, 0.01, 10, 1, 0])
    vg.add("subgoal_interval", [3, 5, 7])#0.01])#, 0.01, 10, 1, 0])

    variants = vg.variants()#randomized=True)

    print("#Experiments: ", len(variants))

    for variant in variants:
        def new_high_network(output_dim, output_nonlinearity):
            return ConvNetwork(
                input_shape=env.observation_space.shape,
                output_dim=output_dim,
                hidden_sizes=(10,),
                conv_filters=(5, 5),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('full', 'full'),
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=output_nonlinearity,
            )


        def new_low_network(output_dim, output_nonlinearity):
            return ConvMergeNetwork(
                input_shape=env.observation_space.shape,
                extra_input_shape=(variant["n_subgoals"],),
                output_dim=output_dim,
                extra_hidden_sizes=(10,),
                hidden_sizes=(10,),
                conv_filters=(5, 5),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('full', 'full'),
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=output_nonlinearity,
            )

        policy = SubgoalPolicy(
            env_spec=env.spec,
            high_policy_cls=CategoricalMLPPolicy,
            high_policy_args=dict(prob_network=new_high_network(variant["n_subgoals"], NL.softmax)),
            low_policy_cls=CategoricalMLPPolicy,
            low_policy_args=dict(prob_network=new_low_network(env.action_space.flat_dim, NL.softmax)),
            subgoal_space=Discrete(variant["n_subgoals"]),
            subgoal_interval=variant["subgoal_interval"],
        )

        baseline = SubgoalBaseline(
            env_spec=env.spec,
            high_baseline=ZeroBaseline(policy.high_env_spec),
            low_baseline=ZeroBaseline(policy.low_env_spec),
        )

        mi_evaluator = StateBasedValueMIEvaluator(
            env=env,
            policy=policy,
            discount=0.99,
            state_regressor_args=dict(
                # use_trust_region=True,
                # optimizer=ConjugateGradientOptimizer(),
                mean_network=new_high_network(1, NL.identity),
            ),
            state_goal_regressor_args=dict(
                # use_trust_region=True,
                # optimizer=ConjugateGradientOptimizer(),
                mean_network=new_low_network(1, NL.identity),
            ),
        )

        algo = BatchHRL(
            env=env,
            policy=policy,
            baseline=baseline,
            bonus_evaluator=mi_evaluator,
            batch_size=batch_size,
            mi_coeff=variant["mi_coeff"],
            max_path_length=100,
            n_itr=100,
            high_algo=TRPO(
                env=env,
                policy=policy.high_policy,
                baseline=baseline.high_baseline,
                discount=0.99,
                step_size=0.01,
            ),
            low_algo=TRPO(
                env=env,
                policy=policy.low_policy,
                baseline=baseline.low_baseline,
                discount=0.99,
                step_size=0.01,
            ),
        )

        # tasks.append(dict(
        #     stub_method_call=algo.train(),
        #     seed=variant["seed"],
        #     n_parallel=1,
        # ))

        instrument.run_experiment_lite(
            algo.train(),
            exp_prefix="hrl_seaquest_8x8_no_trust_region",
            snapshot_mode="last",
            mode="lab_kube",
        )

        # sys.exit(0)

    # n_machines = 10
    # n_runs_per_machine = int(np.ceil(len(tasks) * 1.0 / n_machines))
    #
    # for idx in xrange(0, len(tasks), n_runs_per_machine):
    #     machine_tasks = tasks[idx:idx + n_runs_per_machine]
    #     instrument.run_experiment_lite(
    #         batch_tasks=machine_tasks,
    #         exp_prefix="hrl_seaquest",
    #         snapshot_mode="last",
    #         mode="local",
    #     )
    #     sys.exit(0)

else:
    for seed in [11, 21, 31, 41, 51]:
        for size in [4]:  # , 15]:
            guided_obs_size = size + size + 2
            env = SeaquestGridWorldEnv(
                size=size,
                n_bombs=size / 2,
                guided_observation=True,
            )

            network = ConvMergeNetwork(
                input_shape=env.observation_space.components[0].shape,
                extra_input_shape=(guided_obs_size,),
                output_dim=env.action_space.n,
                hidden_sizes=(5,),
                conv_filters=(5, 5),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('full', 'full'),
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=NL.softmax,
            )
            policy = CategoricalMLPPolicy(env_spec=env.spec, prob_network=network)
            baseline = ZeroBaseline(env_spec=env.spec)

            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=4000,
                max_path_length=100,
                n_itr=1000,
            )

            instrument.run_experiment_lite(
                algo.train(),
                exp_prefix="seaquest",
                snapshot_mode="last",
                seed=seed,
                n_parallel=1,
            )
            # sys.exit(0)
