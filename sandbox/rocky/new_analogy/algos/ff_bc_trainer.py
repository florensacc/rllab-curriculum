from rllab.core.serializable import Serializable
import tensorflow as tf

from rllab.envs.base import Env
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np
from rllab.misc import logger


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

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(name="obs", extra_dims=1)
        action_var = self.env.action_space.new_tensor_variable(name="action", extra_dims=1)

        dist_info = self.policy.dist_info_sym(obs_var=obs_var)

        # minimize the M-projection KL divergence

        loss = tf.reduce_mean(tf.square(dist_info["mean"] - action_var))

        self.optimizer = FirstOrderOptimizer(
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=dict(learning_rate=self.learning_rate),
            max_epochs=self.n_epochs,
            batch_size=self.batch_size,
            tolerance=None,
            verbose=True,
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
            sampler = VectorizedSampler(
                env=self.env,
                policy=self.policy,
                n_envs=100
            )
            logger.log("Starting worker...")
            sampler.start_worker()
            logger.log("Worker started")
            sess.run(tf.assign(self.policy._l_std_param.param, [-10] * self.env.action_dim))

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
                    max_path_length=100,
                    batch_size=10000,
                    max_n_trajs=100
                )
                logger.record_tabular('AverageExpertReward', np.mean([np.sum(p["rewards"]) for p in self.train_paths]))
                logger.record_tabular('AveragePolicyReward', np.mean([np.sum(p["rewards"]) for p in pol_paths]))
                logger.record_tabular('SuccessRate', np.mean([p["rewards"][-1] >= 4 for p in pol_paths]))
            logger.dump_tabular()

            logger.save_itr_params(itr, dict(env=self.env, policy=self.policy))

        self.optimizer.optimize(inputs=[train_observations, train_actions], callback=cb)

        if self.evaluate_performance:
            sampler.shutdown_worker()
        if session_created:
            sess.__exit__(None, None, None)
