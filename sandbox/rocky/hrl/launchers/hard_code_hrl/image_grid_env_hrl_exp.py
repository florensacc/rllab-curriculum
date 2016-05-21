from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.envs.image_grid_env import ImageGridEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.core.network import ConvNetwork
from rllab.algos.trpo import TRPO
from rllab.algos.nop import NOP
from rllab.spaces.discrete import Discrete
from rllab.misc import instrument
from rllab.misc.instrument import variant
import lasagne.nonlinearities as NL
import sys
from sandbox.rocky.hrl.policies.subgoal_policy import SubgoalPolicy
from sandbox.rocky.hrl.baselines.subgoal_baseline import SubgoalBaseline
from sandbox.rocky.hrl.core.network import ActionConditionedConvDeconvNetwork, ConvMergeNetwork
from sandbox.rocky.hrl.algos.batch_hrl import BatchHRL
from sandbox.rocky.hrl.mi_evaluator.state_based_mi_evaluator import StateBasedMIEvaluator
from sandbox.rocky.hrl.mi_evaluator.state_based_value_mi_evaluator import StateBasedValueMIEvaluator
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor

instrument.stub(globals())

env = ImageGridEnv(size=4, subgoal_interval=2, action_interval=1)


class VG(instrument.VariantGenerator):
    @variant
    def seed(self):
        return [1, 11, 21]

    @variant
    def subgoal_interval(self):
        return [2]

    @variant
    def n_subgoals(self):
        yield 5
        # return [13, 20, 30]

    @variant
    def mi_coeff(self):
        return [1.0]

    @variant
    def reward_coeff(self):
        return [0.0]

    @variant
    def mi_use_trust_region(self):
        yield True
        # return [True, False]

    @variant
    def mi_step_size(self, mi_use_trust_region):
        yield 1.0
        # if mi_use_trust_region:
        #     return [10.0, 0.1, 0.01, 1.0]
        # else:
        #     return [None]

    @variant
    def mi_evaluator_cls(self):
        yield StateBasedMIEvaluator
        # return [StateBasedMIEvaluator, StateBasedValueMIEvaluator]

    @variant
    def mi_regressor_network(self, mi_evaluator_cls, n_subgoals):
        if mi_evaluator_cls == StateBasedMIEvaluator:
            yield ActionConditionedConvDeconvNetwork(
                image_shape=env.observation_space.shape,
                action_dim=n_subgoals,
                conv_filters=(5, 5),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('full', 'full'),
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=NL.softmax,
                embedding_dim=10,
            )
        elif mi_evaluator_cls == StateBasedValueMIEvaluator:
            yield ConvMergeNetwork(
                input_shape=env.observation_space.shape,
                extra_input_shape=(n_subgoals,),
                output_dim=1,
                extra_hidden_sizes=(10,),
                hidden_sizes=(10,),
                conv_filters=(5, 5),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('full', 'full'),
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=NL.identity,
            )
        else:
            raise ValueError

    @variant
    def mi_optimizer(self, mi_use_trust_region):
        if mi_use_trust_region:
            return [ConjugateGradientOptimizer()]
        else:
            return [FirstOrderOptimizer(max_epochs=5)]

    @variant
    def mi_regressor_args(self, mi_step_size, mi_use_trust_region, mi_optimizer, mi_regressor_network):
        args = dict(
            optimizer=mi_optimizer,
            use_trust_region=mi_use_trust_region,
            mean_network=mi_regressor_network,
        )
        if mi_use_trust_region:
            args["step_size"] = mi_step_size
        yield args

    @variant
    def algo(self, n_subgoals, subgoal_interval, mi_evaluator_cls, mi_regressor_args, mi_coeff, reward_coeff):
        policy = SubgoalPolicy(
            env_spec=env.spec,
            high_policy_cls=CategoricalMLPPolicy,
            high_policy_args=dict(
                prob_network=ConvNetwork(
                    input_shape=env.observation_space.shape,
                    output_dim=n_subgoals,
                    hidden_sizes=(10,),
                    conv_filters=(5, 5),
                    conv_filter_sizes=(1, 1),
                    conv_strides=(1, 1),
                    conv_pads=('full', 'full'),
                    hidden_nonlinearity=NL.tanh,
                    output_nonlinearity=NL.softmax,
                ),
            ),
            low_policy_cls=CategoricalMLPPolicy,
            low_policy_args=dict(
                prob_network=ConvMergeNetwork(
                    input_shape=env.observation_space.shape,
                    extra_input_shape=(n_subgoals,),
                    output_dim=env.action_space.n,
                    extra_hidden_sizes=(10,),
                    hidden_sizes=(10,),
                    conv_filters=(5, 5),
                    conv_filter_sizes=(3, 3),
                    conv_strides=(1, 1),
                    conv_pads=('full', 'full'),
                    hidden_nonlinearity=NL.tanh,
                    output_nonlinearity=NL.softmax,
                )

            ),
            subgoal_space=Discrete(n_subgoals),
            subgoal_interval=subgoal_interval,
        )
        baseline = SubgoalBaseline(
            env_spec=env.spec,
            high_baseline=ZeroBaseline(env_spec=policy.high_env_spec),
            low_baseline=ZeroBaseline(env_spec=policy.low_env_spec),
        )

        mi_evaluator = mi_evaluator_cls(
            env=env,
            policy=policy,
            regressor_cls=GaussianMLPRegressor,
            regressor_args=mi_regressor_args,
        )

        yield BatchHRL(
            env=env,
            policy=policy,
            baseline=baseline,
            bonus_evaluator=mi_evaluator,
            batch_size=10000,
            mi_coeff=mi_coeff,
            reward_coeff=reward_coeff,
            max_path_length=100,
            n_itr=100,
            high_algo=NOP(
                env=env,
                policy=policy.high_policy,
                baseline=baseline.high_baseline,
                discount=0.99,
            ),
            low_algo=TRPO(
                env=env,
                policy=policy.low_policy,
                baseline=baseline.low_baseline,
                discount=0.99,
                step_size=0.01,
            ),
        )


vg = VG()

variants = vg.variants(randomized=False)

print("# Experiments: %d" % len(variants))

# Experiment: test if a larger grid needs more time to train

for v in variants:
    instrument.run_experiment_lite(
        v["algo"].train(),
        n_parallel=1,
        exp_prefix="image_grid_hrl_no_ext_exp",
        mode="local",
        snapshot_mode="last",
        seed=v["seed"],
    )
    # sys.exit(0)
