from rllab.core.serializable import Serializable
import tensorflow as tf

from rllab.envs.base import Env
from sandbox.rocky.chainer.optimizers.tbptt_optimizer import TBPTTOptimizer
from sandbox.rocky.chainer.misc import tensor_utils
from sandbox.rocky.chainer.samplers.vectorized_sampler import VectorizedSampler
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
            opt_batch_size=128,
            opt_n_steps=None,
            learning_rate=1e-3,
            evaluate_performance=True,
            eval_samples=10000,
            eval_horizon=100,
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
        self.opt_batch_size = opt_batch_size
        self.opt_n_steps = opt_n_steps
        self.learning_rate = learning_rate
        self.evaluate_performance = evaluate_performance
        self.eval_samples = eval_samples
        self.eval_horizon = eval_horizon

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(name="obs", extra_dims=2)
        action_var = self.env.action_space.new_tensor_variable(name="action", extra_dims=2)

        rnn_network = self.policy.head_network
        state_dim = rnn_network.state_dim
        recurrent_layer = rnn_network.recurrent_layer
        state_init_param = rnn_network.state_init_param

        state_var = tf.placeholder(tf.float32, (None, state_dim), "state")
        recurrent_state_output = dict()

        minibatch_dist_info_vars = self.policy.dist_info_sym(
            obs_var, state_info_vars=dict(),
            recurrent_state={recurrent_layer: state_var},
            recurrent_state_output=recurrent_state_output,
        )

        state_output = recurrent_state_output[recurrent_layer]

        loss = tf.reduce_mean(tf.square(action_var - minibatch_dist_info_vars["mean"]))

        final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]

        self.optimizer = TBPTTOptimizer(
            tf_optimizer_cls=tf.train.AdamOptimizer,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            batch_size=self.opt_batch_size,
            n_steps=self.opt_n_steps,
            tolerance=None,
            verbose=True,
        )

        self.optimizer.update_opt(
            loss=loss,
            target=self.policy,
            inputs=[obs_var, action_var],
            rnn_init_state=state_init_param,
            rnn_state_input=state_var,
            rnn_final_state=final_state,
        )

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
                n_envs=max(1, min(100, int(np.ceil(self.eval_samples / self.eval_horizon))))
            )
            logger.log("Starting worker...")
            sampler.start_worker()
            logger.log("Worker started")
            sess.run(tf.assign(self.policy.l_log_std.param, [-10] * self.env.action_dim))

        max_path_length = np.max([len(p["observations"]) for p in self.train_paths + self.test_paths])

        train_obs = [path["observations"] for path in self.train_paths]
        train_obs = tensor_utils.pad_tensor_n(train_obs, max_path_length)
        train_actions = [path["actions"] for path in self.train_paths]
        train_actions = tensor_utils.pad_tensor_n(train_actions, max_path_length)

        test_obs = [path["observations"] for path in self.test_paths]
        test_obs = tensor_utils.pad_tensor_n(test_obs, max_path_length)
        test_actions = [path["actions"] for path in self.test_paths]
        test_actions = tensor_utils.pad_tensor_n(test_actions, max_path_length)

        def cb(itr, loss, best_loss, n_no_improvements, learning_rate, diagnostics):
            test_paths = sampler.obtain_samples(
                itr=itr,
                max_path_length=self.eval_horizon,
                batch_size=self.eval_samples,
                max_n_trajs=int(np.ceil(self.eval_samples / self.eval_horizon)),
            )

            logger.record_tabular('Epoch', itr)
            logger.record_tabular('TrainLoss', loss)
            test_loss, _ = self.optimizer.loss_diagnostics([test_obs, test_actions])
            logger.record_tabular('TestLoss', test_loss)
            logger.record_tabular('AverageExpertReward', np.mean([np.sum(p["rewards"]) for p in self.train_paths]))
            logger.record_tabular('AveragePolicyReward', np.mean([np.sum(p["rewards"]) for p in test_paths]))
            logger.record_tabular('SuccessRate', np.mean([p["rewards"][-1] >= 4 for p in test_paths]))
            logger.record_tabular('BestTrainLoss', best_loss)
            logger.record_tabular('NNoImprovements', n_no_improvements)
            logger.record_tabular('LearningRate', learning_rate)
            logger.dump_tabular()

            logger.save_itr_params(itr, dict(env=self.env, policy=self.policy))

            return True

        self.optimizer.optimize(inputs=[train_obs, train_actions], callback=cb)

        if self.evaluate_performance:
            sampler.shutdown_worker()
        if session_created:
            sess.__exit__(None, None, None)
