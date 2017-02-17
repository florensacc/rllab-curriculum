import numpy as np

from rllab.misc import special
from rllab.misc import ext
from rllab.misc import logger
from sandbox.rocky.benchmark.sampling import sample_rollouts
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf

from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp


class NAF(object):
    def __init__(
            self,
            env,
            qf,
            # policy,
            # baseline,
            batch_size=1000,
            max_path_length=500,
            n_itr=500,
            step_size=0.01,
            discount=0.995,
            gae_lambda=0.97,
            n_envs=10,
            parallel=False,
            adv_normalize_momemtum=0.9,
            optimizer=None,
    ):
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.n_itr = n_itr
        self.step_size = step_size
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.n_envs = n_envs
        self.parallel = parallel
        self.adv_normalize_momentum = adv_normalize_momemtum
        self.adv_mean = 0.
        self.adv_std = 1.
        if optimizer is None:
            optimizer = ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
        self.optimizer = optimizer

    def init_opt(self):
        # Initialize optimization
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        mean_kl = tf.reduce_mean(kl)
        surr_loss = - tf.reduce_mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

    def process_samples(self, itr, paths):
        if hasattr(self.baseline, "predict_n"):
            all_path_baselines = self.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.baseline.predict(path) for path in paths]
        for idx, path in enumerate(paths):
            path_baselines = all_path_baselines[idx]
            assert len(path_baselines) == len(path["rewards"]) + 1
            deltas = path["rewards"] + self.discount * path_baselines[1:] - path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            if path["dones"][-1]:
                v_last = 0
            else:
                v_last = path_baselines[-1]
            path["baselines"] = path_baselines
            path["returns"] = special.discount_cumsum(
                np.append(path["rewards"], v_last),
                self.discount
            )

        observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
        actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
        rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
        advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        # normalize advantages
        if itr == 0:
            self.adv_mean = np.mean(advantages)
            self.adv_std = np.std(advantages)
        else:
            m = self.adv_normalize_momentum
            self.adv_mean = m * self.adv_mean + (1 - m) * np.mean(advantages)
            self.adv_std = m * self.adv_std + (1 - m) * np.std(advantages)

        advantages = (advantages - self.adv_mean) / (self.adv_std + 1e-8)

        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
        )

    def optimize_policy(self, paths, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)

    def optimize_baseline(self, paths, samples_data):
        self.baseline.fit(paths)

    def log_diagnostics(self, paths, samples_data):
        baselines = []
        discounted_returns = []
        for idx, path in enumerate(paths):
            baselines.append(path["baselines"][:-1])
            discounted_returns.append(path["returns"][:-1])
        ent = np.mean(self.policy.distribution.entropy(samples_data["agent_infos"]))
        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(discounted_returns)
        )
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            env=self.env,
            policy=self.policy,
            baseline=self.baseline,
        )

    def train(self):
        self.init_opt()
        with tf.Session() as sess:
            tensor_utils.initialize_new_variables(sess=sess)
            n_samples = 0
            rollout_itr = sample_rollouts(
                env=self.env, policy=self.policy, batch_size=self.batch_size, max_path_length=self.max_path_length,
                n_envs=self.n_envs, parallel=self.parallel)
            for itr in range(self.n_itr):
                logger.record_tabular('Itr', itr)
                paths = next(rollout_itr)
                n_samples += np.sum([len(p["rewards"]) for p in paths])
                logger.record_tabular('NSamples', n_samples)
                samples_data = self.process_samples(itr, paths)
                self.optimize_baseline(paths, samples_data)
                self.optimize_policy(paths, samples_data)
                self.log_diagnostics(paths, samples_data)
                params = self.get_itr_snapshot(itr, samples_data)
                logger.save_itr_params(itr, params, use_cloudpickle=True)
                logger.dump_tabular(with_prefix=False)
                if itr + 1 >= self.n_itr:
                    break
