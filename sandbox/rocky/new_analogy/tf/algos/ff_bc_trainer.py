import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.misc import logger
from sandbox.rocky.new_analogy.tf.policies.deterministic_policy import DeterministicPolicy
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler


class Trainer(Serializable):
    def __init__(
            self,
            env: Env,
            policy,
            paths,
            threshold=None,
            train_ratio=0.9,
            n_epochs=100,
            batch_size=128,
            learning_rate=1e-3,
            evaluate_performance=True,
            max_path_length=100,
            n_eval_trajs=100,
            eval_batch_size=10000,
            n_eval_envs=100,
            n_passes_per_epoch=1,
            n_slices=1,
            learn_std=False,
    ):
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy = policy
        if threshold is not None:
            paths = [p for p in paths if p["rewards"][-1] >= threshold]

        n_train = int(train_ratio * len(paths))

        np.random.shuffle(paths)

        train_paths = paths[:n_train]
        test_paths = paths[n_train:]
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.evaluate_performance = evaluate_performance
        self.max_path_length = max_path_length
        self.n_eval_trajs = n_eval_trajs
        self.eval_batch_size = eval_batch_size
        self.n_eval_envs = n_eval_envs
        self.n_passes_per_epoch = n_passes_per_epoch
        self.n_slices = n_slices
        self.learn_std = learn_std

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(name="obs", extra_dims=1)
        action_var = self.env.action_space.new_tensor_variable(name="action", extra_dims=1)

        dist_info = self.policy.dist_info_sym(obs_var=obs_var)

        if self.learn_std:
            loss = tf.reduce_mean(-self.policy.distribution.log_likelihood_sym(action_var, dist_info))
        else:
            sqrdiff = tf.square(dist_info["mean"] - action_var)
            loss = tf.reduce_mean(sqrdiff)

        self.optimizer = FirstOrderOptimizer(
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=dict(learning_rate=self.learning_rate),
            max_epochs=self.n_epochs,
            batch_size=self.batch_size,
            tolerance=None,
            verbose=True,
            n_passes_per_epoch=self.n_passes_per_epoch,
            n_slices=self.n_slices,
        )

        self.optimizer.update_opt(loss=loss, target=self.policy, inputs=[obs_var, action_var])

    def train(self, sess=None):
        self.init_opt()

        session_created = False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
            session_created = True
        # Only initialize variables that have not been initialized

        tensor_utils.initialize_new_variables(sess=sess)

        if self.evaluate_performance:
            if self.learn_std:
                sampler = VectorizedSampler(
                    env=self.env,
                    policy=self.policy,#DeterministicPolicy(env_spec=self.env.spec, wrapped_policy=self.policy),
                    n_envs=self.n_eval_envs
                )
            else:
                sampler = VectorizedSampler(
                    env=self.env,
                    policy=DeterministicPolicy(env_spec=self.env.spec, wrapped_policy=self.policy),
                    n_envs=self.n_eval_envs
                )
            logger.log("Starting worker...")
            sampler.start_worker()
            logger.log("Worker started")

        train_observations = np.concatenate([p["observations"] for p in self.train_paths])
        train_actions = np.concatenate([p["actions"] for p in self.train_paths])
        if len(self.test_paths) > 0:
            test_observations = np.concatenate([p["observations"] for p in self.test_paths])
            test_actions = np.concatenate([p["actions"] for p in self.test_paths])

        def cb(itr, loss, elapsed, params):
            logger.log("Epoch {}".format(itr))
            logger.record_tabular('Epoch', itr)
            logger.record_tabular('TrainLoss', loss)
            if len(self.test_paths) > 0:
                test_loss = self.optimizer.loss([test_observations, test_actions])
            else:
                test_loss = np.nan
            logger.record_tabular('TestLoss', test_loss)

            if self.evaluate_performance:
                pol_paths = sampler.obtain_samples(
                    itr=itr,
                    max_path_length=self.max_path_length,
                    batch_size=self.eval_batch_size,
                    max_n_trajs=self.n_eval_trajs
                )
                logger.record_tabular('AverageExpertReward', np.mean([np.sum(p["rewards"]) for p in self.train_paths]))
                logger.record_tabular_misc_stat('Return', [np.sum(p["rewards"]) for p in pol_paths])
                logger.record_tabular_misc_stat('FinalReward', np.asarray([p["rewards"][-1] for p in pol_paths]))
                self.env.log_diagnostics(pol_paths)

                # import ipdb; ipdb.set_trace()

            logger.dump_tabular()
            # import ipdb; ipdb.set_trace()

            logger.log("Saving params...")
            logger.save_itr_params(itr, dict(env=self.env, policy=self.policy))
            logger.log("Saved")

        self.optimizer.optimize(inputs=[train_observations, train_actions], callback=cb)

        if self.evaluate_performance:
            sampler.shutdown_worker()
        if session_created:
            sess.__exit__(None, None, None)
