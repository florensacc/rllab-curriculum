


from rllab.algos.base import RLAlgorithm
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.hrl.envs.supervised_env import SupervisedEnv
from sandbox.rocky.hrl.bonus_evaluators.discrete_bonus_evaluator import DiscreteBonusEvaluator, MODES
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.sampler import parallel_sampler
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc import tensor_utils
from rllab.misc import special
from rllab.core.network import MLP
from rllab.core.parameterized import Parameterized
import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano.tensor as TT
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        return [param for comp in self.components for param in comp.get_params_internal(**tags)]


class SupervisedRecurrentPolopt(RLAlgorithm):
    def __init__(self,
                 env,
                 policy,
                 recog_network=None,
                 recog_hidden_sizes=(32, 32),
                 optimizer=None,
                 discount=0.99,
                 n_itr=100,
                 bonus_mode=MODES.MODE_MI_FEUDAL_SYNC,
                 bonus_coeff=1.,
                 bottleneck_coeff=1.,
                 learning_rate=1e-3,
                 n_gradient_updates=1,
                 n_on_policy_samples=10000,
                 max_path_length=100):
        """
        :type env: SupervisedEnv
        :type policy: StochasticGRUPolicy
        :type optimizer: LbfgsOptimizer
        """
        self.env = env
        self.policy = policy
        if optimizer is None:
            optimizer = FirstOrderOptimizer(batch_size=None, learning_rate=learning_rate, max_epochs=n_gradient_updates)
        if recog_network is None:
            recog_network = MLP(
                input_shape=(env.observation_space.flat_dim + env.action_space.flat_dim,),
                output_dim=policy.n_subgoals,
                hidden_sizes=recog_hidden_sizes,
                hidden_nonlinearity=NL.tanh,
                output_nonlinearity=NL.softmax,
            )
        self.recog_network = recog_network
        self.bonus_evaluator = DiscreteBonusEvaluator(
            env_spec=env.spec,
            policy=policy,
            # doesn't really matter which mode we pick here; we're only using it for logging
            mode=bonus_mode,
            regressor_args=dict(
                use_trust_region=False,
            ),
            bonus_coeff=bonus_coeff,
            bottleneck_coeff=bottleneck_coeff,
        )

        self.baseline = LinearFeatureBaseline(env_spec=env.spec)
        self.optimizer = optimizer
        self.discount = discount
        self.n_itr = n_itr
        self.n_on_policy_samples = n_on_policy_samples
        self.max_path_length = max_path_length
        self.bonus_coeff = bonus_coeff

    def init_opt(self):

        # *******************************************
        #      Supervised part of the objective
        # *******************************************

        training_paths = self.env.generate_training_paths()

        obs_var = self.env.observation_space.new_tensor_variable(
            name="obs",
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            name="action",
            extra_dims=1
        )

        start_mask_var = TT.ivector(name="start_mask")
        recog_hidden_probs = L.get_output(
            self.recog_network.output_layer,
            {self.recog_network.input_layer: TT.concatenate([obs_var, action_var], axis=1)}
        )
        srng = RandomStreams()

        recog_hidden_states = srng.multinomial(pvals=recog_hidden_probs)
        recog_prev_hiddens = TT.concatenate([
            TT.zeros((1, self.policy.n_subgoals)),
            recog_hidden_states[1:]
        ], axis=0)
        recog_prev_hiddens = TT.set_subtensor(
            recog_prev_hiddens[start_mask_var.nonzero()],
            np.eye(self.policy.n_subgoals)[0]
        )

        dist_info_sym = self.policy.dist_info_sym(obs_var, dict(
            hidden_state=recog_hidden_states,
            prev_hidden=recog_prev_hiddens
        ))

        per_logli = self.policy.distribution.log_likelihood_sym(action_var, dist_info_sym)

        recog_ent = self.policy.hidden_dist.entropy_sym(dict(prob=recog_hidden_probs))
        recog_hidden_logli = self.policy.hidden_dist.log_likelihood_sym(recog_hidden_states,
                                                                        dict(prob=recog_hidden_probs))
        loss = TT.mean(- per_logli - recog_ent)

        surr = TT.mean(- per_logli - theano.gradient.zero_grad(per_logli) * recog_hidden_logli - recog_ent)

        all_observations = np.concatenate([p["observations"] for p in training_paths])
        all_actions = np.concatenate([p["actions"] for p in training_paths])
        all_start_ids = np.concatenate([[0], np.cumsum([len(p["observations"]) for p in training_paths])])[:-1]
        all_start_mask = np.zeros((len(all_actions),))
        all_start_mask[all_start_ids] = 1

        self.sup_inputs = [all_observations, all_actions, all_start_mask]

        # *******************************************
        #      Information regularization terms
        # *******************************************
        on_policy_obs_var = self.env.observation_space.new_tensor_variable(
            name="on_policy_obs",
            extra_dims=1
        )
        on_policy_actions_var = self.env.action_space.new_tensor_variable(
            name="on_policy_actions",
            extra_dims=1
        )
        on_policy_state_info_vars = {
            k: TT.matrix(name="on_policy_%s" % k)
            for k in self.policy.state_info_keys
            }
        on_policy_state_info_vars_list = [on_policy_state_info_vars[k] for k in self.policy.state_info_keys]

        on_policy_adv_var = TT.vector(name="on_policy_adv")
        on_policy_dist_info_vars = self.policy.dist_info_sym(on_policy_obs_var, on_policy_state_info_vars)
        action_logli_var = self.policy.log_likelihood_sym(on_policy_actions_var, on_policy_dist_info_vars)
        reg_surr = -TT.mean(action_logli_var * on_policy_adv_var)

        # *******************************************
        #      Configure optimizer
        # *******************************************

        # normalize
        surr = (surr + self.bonus_coeff * reg_surr) / (1 + self.bonus_coeff)
        loss = (loss + self.bonus_coeff * reg_surr) / (1 + self.bonus_coeff)
        # surr = reg_surr
        # loss = reg_surr
        joint_target = JointParameterized([self.policy, self.recog_network])
        grads = TT.grad(surr, joint_target.get_params(trainable=True), disconnected_inputs='ignore')

        all_input_vars = [obs_var, action_var, start_mask_var, on_policy_obs_var, on_policy_actions_var,
                          on_policy_adv_var] + on_policy_state_info_vars_list

        f_recog_ent = ext.compile_function(all_input_vars, TT.mean(recog_ent))
        self.optimizer.update_opt(loss, joint_target, all_input_vars, gradients=grads)
        self.f_recog_ent = f_recog_ent

    def start_worker(self):
        self.env.test_mode()
        parallel_sampler.populate_task(self.env, self.policy)

    def train(self):
        assert not self.policy.use_decision_nodes
        assert not self.policy.random_reset
        self.init_opt()
        self.start_worker()

        for itr in range(self.n_itr):
            paths = parallel_sampler.sample_paths(
                policy_params=self.policy.get_param_values(),
                max_samples=self.n_on_policy_samples,
                max_path_length=self.max_path_length
            )
            self.bonus_evaluator.fit(paths)

            # these need to be computed now since I'm messing with the rewards below
            avg_discounted_return = np.mean(
                [np.sum((self.discount ** np.arange(len(p["rewards"]))) * p["rewards"]) for p in paths]
            )
            avg_return = np.mean(
                [np.sum(p["rewards"]) for p in paths]
            )
            for path in paths:
                path["rewards"] = self.bonus_evaluator.predict(path)
                path_baselines = np.append(self.baseline.predict(path), 0)
                deltas = path["rewards"] + \
                         self.discount * path_baselines[1:] - \
                         path_baselines[:-1]
                path["advantages"] = special.discount_cumsum(deltas, self.discount)
                path["returns"] = special.discount_cumsum(path["rewards"], self.discount)

            on_policy_obs = np.concatenate([p["observations"] for p in paths])
            on_policy_actions = np.concatenate([p["actions"] for p in paths])
            on_policy_advs = np.concatenate([p["advantages"] for p in paths])
            on_policy_agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
            # center the advantages
            on_policy_advs = (on_policy_advs - np.mean(on_policy_advs)) / (np.std(on_policy_advs) + 1e-8)

            # update the baseline
            self.baseline.fit(paths)

            state_info_list = [on_policy_agent_infos[k] for k in self.policy.state_info_keys]

            all_inputs = self.sup_inputs + [on_policy_obs, on_policy_actions, on_policy_advs] + state_info_list

            loss_before = self.optimizer.loss(all_inputs)
            self.optimizer.optimize(all_inputs)
            loss_after = self.optimizer.loss(all_inputs)

            logger.record_tabular("Itr", itr)
            logger.record_tabular("AverageDiscountedReturn", avg_discounted_return)
            logger.record_tabular("AverageReturn", avg_return)

            logger.record_tabular("RecogEntropy", self.f_recog_ent(*all_inputs))
            logger.record_tabular("PolicyEntropy", np.mean(self.policy.entropy(on_policy_agent_infos)))

            logger.record_tabular("LossBefore", loss_before)
            logger.record_tabular("LossAfter", loss_after)

            self.bonus_evaluator.log_diagnostics(paths)

            logger.dump_tabular()
