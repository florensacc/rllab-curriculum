from collections import OrderedDict

from sandbox.rocky.neural_learner.envs.mab_env import VecMAB
from sandbox.rocky.neural_learner.envs.multi_env import VecMultiEnv, MultiEnv
from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
from sandbox.rocky.neural_learner.samplers.vectorized_sampler import VectorizedSampler
# from sandbox.rocky.tf.core.parameterized import suppress_params_loading
from rllab.misc import logger
from rllab.misc import special
import tensorflow as tf
import numpy as np
import tempfile

from sandbox.rocky.tf.envs.base import VecTfEnv, TfEnv
from sandbox.rocky.tf.misc import tensor_utils


class MDGPSTrainer(object):
    def __init__(
            self,
            env,
            policy,
            batch_size,
            max_path_length,
            cache_key,
            eval_batch_size,
            step_size=0.1,
            optimizer=None,
            initial_kl_penalty=1.,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            use_kl_penalty=False,
            min_penalty=1e-3,
            max_penalty=1e6,
            max_backtracks=10,
            backtrack_ratio=0.5,
            lagrangian_penalty=1.,
            n_itr=100,
    ):
        self.env = env
        self.policy = policy
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_path_length = max_path_length
        n_eval_envs = min(100, max(1, int(np.ceil(self.eval_batch_size / self.max_path_length))))
        self.eval_sampler = VectorizedSampler(env=self.env, policy=self.policy, n_envs=n_eval_envs)
        self.cache_key = cache_key
        if optimizer is None:
            optimizer = TBPTTOptimizer()
        self.optimizer = optimizer
        self.step_size = step_size
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.initial_kl_penalty = initial_kl_penalty
        self.max_backtracks = max_backtracks
        self.backtrack_ratio = backtrack_ratio
        self.use_kl_penalty = use_kl_penalty
        self.lagrangian_penalty = lagrangian_penalty
        self.n_itr = n_itr

        self.kl_penalty_var = None
        self.f_increase_penalty = None
        self.f_decrease_penalty = None
        self.f_reset_penalty = None

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            name="obs",
            extra_dims=2,
        )
        dist = self.policy.distribution
        ref_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None, None) + shape, name=k + "_ref")
            for k, shape in dist.dist_info_specs
        }
        ref_dist_info_vars_list = [ref_dist_info_vars[k] for k, _ in dist.dist_info_specs]
        assert len(self.policy.state_info_keys) == 0

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None, None) + shape, name=k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k, _ in dist.dist_info_specs]

        valid_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="valid")

        rnn_network = self.policy.prob_network

        state_var = tf.placeholder(tf.float32, (None, rnn_network.state_dim), "state")

        kl_penalty_var = tf.Variable(
            initial_value=self.initial_kl_penalty,
            dtype=tf.float32,
            name="kl_penalty"
        )

        recurrent_layer = rnn_network.recurrent_layer
        recurrent_state_output = dict()

        minibatch_dist_info_vars = self.policy.dist_info_sym(
            obs_var, state_info_vars=dict(),
            recurrent_state={recurrent_layer: state_var},
            recurrent_state_output=recurrent_state_output,
        )

        state_output = recurrent_state_output[rnn_network.recurrent_layer]
        final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]

        teacher_kl = tf.reduce_sum(
            dist.kl_sym(ref_dist_info_vars, minibatch_dist_info_vars) * valid_var
        ) / tf.reduce_sum(valid_var)

        old_kl = tf.reduce_sum(
            dist.kl_sym(old_dist_info_vars, minibatch_dist_info_vars) * valid_var
        ) / tf.reduce_sum(valid_var)

        if self.use_kl_penalty:
            total_loss = teacher_kl + kl_penalty_var * tf.maximum(old_kl - self.step_size, 0.)
        else:
            total_loss = teacher_kl

        self.optimizer.update_opt(
            loss=total_loss,
            target=self.policy,
            inputs=[obs_var] + ref_dist_info_vars_list + old_dist_info_vars_list + [valid_var],
            rnn_init_state=rnn_network.state_init_param,
            rnn_state_input=state_var,
            rnn_final_state=final_state,
            diagnostic_vars=OrderedDict([
                ("OldKL", old_kl),
                ("TeacherKL", teacher_kl),
            ])
        )

        self.kl_penalty_var = kl_penalty_var

        self.f_increase_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                tf.minimum(kl_penalty_var * self.increase_penalty_factor, self.max_penalty)
            )
        )
        self.f_decrease_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                tf.maximum(kl_penalty_var * self.decrease_penalty_factor, self.min_penalty)
            )
        )
        self.f_reset_penalty = tensor_utils.compile_function(
            inputs=[],
            outputs=tf.assign(
                kl_penalty_var,
                self.initial_kl_penalty
            )
        )

    def obtain_samples(self, itr):
        n_envs = min(100, max(1, int(np.ceil(self.batch_size / self.max_path_length))))
        vec_env = self.env.vec_env_executor(n_envs=n_envs)

        lagrangian_penalty = self.lagrangian_penalty

        class TeacherPolicy(object):
            def __init__(self, vec_env, policy):
                assert isinstance(vec_env, VecTfEnv) and isinstance(vec_env.vec_env, VecMultiEnv)
                vec_mab_env = vec_env.vec_env.vec_env
                assert isinstance(vec_mab_env, VecMAB)
                self.vec_env = vec_env
                self.vec_mab_env = vec_mab_env
                self.policy = policy

            def reset(self, dones):
                self.policy.reset(dones)

            def get_actions(self, observations):
                arm_means = self.vec_mab_env.arm_means
                action_dim = arm_means.shape[-1]
                if itr == 0:
                    best_arms = np.argmax(arm_means, axis=1)
                    prob = special.to_onehot_n(best_arms, dim=action_dim)
                    return best_arms, dict(prob=prob)
                else:
                    _, policy_agent_infos = self.policy.get_actions(observations)
                    best_arms = np.argmax(arm_means, axis=1)
                    teacher_prob = special.to_onehot_n(best_arms, dim=action_dim)
                    policy_prob = policy_agent_infos["prob"]
                    mixture_prob = 0.1 * teacher_prob + 0.9 * policy_prob


                    # teacher_prob = special.softmax(arm_means / lagrangian_penalty + np.log(policy_agent_infos['prob']
                    #                                                                        + 1e-8))
                    arms = special.weighted_sample_n(mixture_prob, np.arange(action_dim))
                    return arms, dict(prob=teacher_prob)

            @property
            def vectorized(self):
                return True

        # generate optimal trajectories
        sampler = VectorizedSampler(
            env=self.env,
            policy=TeacherPolicy(vec_env=vec_env, policy=self.policy),
            n_envs=n_envs,
            vec_env=vec_env
        )
        paths = sampler.obtain_samples(itr, max_path_length=self.max_path_length, batch_size=self.batch_size)
        return paths

    def process_samples(self, itr, paths):
        T = np.max([len(p["observations"]) for p in paths])
        observations = [p["observations"] for p in paths]
        observations = tensor_utils.pad_tensor_n(observations, T)
        valids = [np.ones((len(p["observations"], ))) for p in paths]
        valids = tensor_utils.pad_tensor_n(valids, T)

        dist_infos = self.policy.dist_info(observations, dict())
        dist_info_keys = self.policy.distribution.dist_info_keys
        dist_info_list = [dist_infos[k] for k in dist_info_keys]

        agent_infos = [p["agent_infos"] for p in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list(
            [tensor_utils.pad_tensor_dict(p, T) for p in agent_infos]
        )
        ref_dist_info_list = [agent_infos[k] for k in dist_info_keys]

        all_inputs = [observations] + ref_dist_info_list + dist_info_list + [valids]

        average_return = np.mean([np.sum(p["rewards"]) for p in paths])
        logger.record_tabular("TeacherAverageReturn", average_return)

        return all_inputs

    def optimize_policy(self, itr, samples_data):
        opt_inputs = samples_data#list(samples_data.values())[:-1]

        prev_params = self.policy.get_param_values(trainable=True)

        best_teacher_kl = None
        best_params = None

        # logger.log("Evaluating loss before")
        # loss_before, diagnostics = self.optimizer.loss_diagnostics(opt_inputs)
        #
        # old_kl_before = diagnostics["OldKL"]
        # teacher_kl_before = diagnostics["TeacherKL"]

        losses = []
        old_kls = []
        teacher_kls = []

        def itr_callback(itr, loss, learning_rate, best_loss, n_no_improvements, diagnostics, *args, **kwargs):
            nonlocal best_teacher_kl
            nonlocal best_params

            logger.log("Loss: {0}".format(loss))
            teacher_kl = diagnostics["TeacherKL"]
            mean_kl = diagnostics["OldKL"]
            if mean_kl <= self.step_size:
                if best_teacher_kl is None or teacher_kl < best_teacher_kl:
                    best_teacher_kl = teacher_kl
                    best_params = self.policy.get_param_values(trainable=True)

            for k, v in diagnostics.items():
                logger.log("{0}: {1}".format(k, v))

            losses.append(loss)
            old_kls.append(mean_kl)
            teacher_kls.append(teacher_kl)

            return True

        self.optimizer.optimize(inputs=opt_inputs, callback=itr_callback)
        # loss, diagnostics = self.optimizer.loss_diagnostics(opt_inputs)
        #
        # old_kl_after = diagnostics["OldKL"]
        # teacher_kl = diagnostics["TeacherKL"]

        # n_trials = 0
        # step_size = 1.
        # now_params = self.policy.get_param_values(trainable=True)
        # while old_kl_after > self.step_size and n_trials < self.max_backtracks:
        #     step_size *= self.backtrack_ratio
        #     self.policy.set_param_values(
        #         (1 - step_size) * prev_params + step_size * now_params,
        #         trainable=True
        #     )
        #     loss, diagnostics = self.optimizer.loss_diagnostics(opt_inputs)
        #     old_kl_after = diagnostics["OldKL"]
        #     teacher_kl = diagnostics["TeacherKL"]
        #     logger.log("After shrinking step, loss = %f, OldKL = %f, TeacherKL = %f" % (loss, old_kl_after, teacher_kl))
        #
        # if best_teacher_kl is None or teacher_kl < best_teacher_kl:
        #     best_params = self.policy.get_param_values(trainable=True)

        # self.policy.set_param_values(best_params, trainable=True)

        # loss, diagnostics = self.optimizer.loss_diagnostics(opt_inputs)
        logger.record_tabular("Loss|FirstEpoch", losses[0])
        logger.record_tabular("Loss|LastEpoch", losses[-1])
        logger.record_tabular("OldKL|FirstEpoch", old_kls[0])
        logger.record_tabular("OldKL|LastEpoch", old_kls[-1])
        logger.record_tabular("TeacherKL|FirstEpoch", teacher_kls[0])
        logger.record_tabular("TeacherKL|LastEpoch", teacher_kls[-1])

    def train(self):
        self.init_opt()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            self.eval_sampler.start_worker()

            for itr in range(self.n_itr):
                logger.record_tabular("Itr", itr)
                logger.log("Obtaining samples")
                paths = self.obtain_samples(itr)
                logger.log("Processing samples")
                samples_data = self.process_samples(itr, paths)
                logger.log("Optimizing policy")
                self.optimize_policy(itr, samples_data)
                logger.log("Evaluating policy performance")
                eval_paths = self.eval_sampler.obtain_samples(itr, max_path_length=self.max_path_length,
                                                  batch_size=self.eval_batch_size)
                policy_average_return = np.mean([np.sum(p["rewards"]) for p in eval_paths])
                logger.record_tabular("PolicyAverageReturn", policy_average_return)
                logger.dump_tabular()
