from rllab.core.serializable import Serializable
import tensorflow as tf

from rllab.envs.base import Env
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np
from rllab.misc import logger


class EnsemblePolicy(Parameterized, Serializable):
    def __init__(self, policy_list):
        Serializable.quick_init(self, locals())
        self.policy_list = policy_list
        Parameterized.__init__(self)

    def get_params_internal(self, **tags):
        return sum([p.get_params(**tags) for p in self.policy_list], [])

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        actions = [p.get_actions(observations)[0] for p in self.policy_list]
        return np.median(np.asarray(actions), axis=0), dict()

    def reset(self):
        for p in self.policy_list:
            p.reset()


class Trainer(Serializable):
    def __init__(
            self,
            env: Env,
            policy_list,
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
    ):
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy_list = policy_list
        self.policy = EnsemblePolicy(policy_list)
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

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(name="obs", extra_dims=1)
        action_var = self.env.action_space.new_tensor_variable(name="action", extra_dims=1)

        loss = 0

        for policy in self.policy_list:
            dist_info = policy.dist_info_sym(obs_var=obs_var)
            # minimize the M-projection KL divergence
            loss += tf.reduce_mean(tf.square(dist_info["mean"] - action_var))

        loss /= len(self.policy_list)

        self.optimizer = FirstOrderOptimizer(
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=dict(learning_rate=self.learning_rate),
            max_epochs=self.n_epochs,
            batch_size=self.batch_size,
            tolerance=None,
            verbose=True,
            n_slices=10,
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
                n_envs=self.n_eval_envs
            )
            logger.log("Starting worker...")
            sampler.start_worker()
            logger.log("Worker started")
            for policy in self.policy_list:
                sess.run(tf.assign(policy._l_std_param.param, [-10] * self.env.action_dim))

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
                logger.record_tabular('AveragePolicyReward', np.mean([np.sum(p["rewards"]) for p in pol_paths]))
                logger.record_tabular('SuccessRate', np.mean([p["rewards"][-1] >= 4 for p in pol_paths]))
            logger.dump_tabular()

            logger.save_itr_params(itr, dict(env=self.env, policy=self.policy))

        self.optimizer.optimize(inputs=[train_observations, train_actions], callback=cb)

        if self.evaluate_performance:
            sampler.shutdown_worker()
        if session_created:
            sess.__exit__(None, None, None)
