from sandbox.rocky.neural_learner.optimizers.tbptt_optimizer import TBPTTOptimizer
from sandbox.rocky.neural_learner.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.s3.resource_manager import resource_manager
from rllab.misc import logger
import tensorflow as tf
import numpy as np
import tempfile

from sandbox.rocky.tf.misc import tensor_utils


class SupervisedTrainer(object):
    def __init__(
            self,
            env,
            policy,
            oracle_policy,
            batch_size,
            max_path_length,
            cache_key,
            eval_batch_size,
            train_ratio=0.9,
            optimizer=None,
    ):
        self.env = env
        self.policy = policy
        self.oracle_policy = oracle_policy
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_path_length = max_path_length
        n_envs = min(100, max(1, int(np.ceil(self.batch_size / self.max_path_length))))
        self.sampler = VectorizedSampler(env=self.env, policy=self.oracle_policy, n_envs=n_envs)
        n_eval_envs = min(100, max(1, int(np.ceil(self.eval_batch_size / self.max_path_length))))
        self.eval_sampler = VectorizedSampler(env=self.env, policy=self.policy, n_envs=n_eval_envs)
        self.cache_key = cache_key
        if optimizer is None:
            optimizer = TBPTTOptimizer()
        self.optimizer = optimizer
        self.train_ratio = train_ratio

    def gen_training_data(self):
        paths = None

        def _mkfile():
            nonlocal paths
            paths = self.sampler.obtain_samples(itr=0, max_path_length=self.max_path_length, batch_size=self.batch_size)
            f = tempfile.NamedTemporaryFile()
            f.close()
            file_name = f.name + ".npz"
            np.savez_compressed(file_name, paths=np.asarray(paths))
            resource_manager.register_file(resource_name=self.cache_key, file_name=file_name)

        file_path = resource_manager.get_file(resource_name=self.cache_key, mkfile=_mkfile)

        if paths is None:
            with open(file_path, "rb") as f:
                paths = list(np.load(f)["paths"])
        return paths

    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            name="obs",
            extra_dims=2,
        )
        action_var = self.env.action_space.new_tensor_variable(
            name="action",
            extra_dims=2
        )
        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None, None) + shape, name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        valid_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="valid")

        rnn_network = self.policy.prob_network

        state_var = tf.placeholder(tf.float32, (None, rnn_network.state_dim), "state")

        recurrent_layer = rnn_network.recurrent_layer
        recurrent_state_output = dict()

        minibatch_dist_info_vars = self.policy.dist_info_sym(
            obs_var, state_info_vars,
            recurrent_state={recurrent_layer: state_var},
            recurrent_state_output=recurrent_state_output,
        )

        state_output = recurrent_state_output[rnn_network.recurrent_layer]
        final_state = tf.reverse(state_output, [False, True, False])[:, 0, :]

        logli = self.policy.distribution.log_likelihood_sym(action_var, minibatch_dist_info_vars)

        loss = - tf.reduce_sum(logli * valid_var) / tf.reduce_sum(valid_var)

        self.optimizer.update_opt(
            loss=loss,
            target=self.policy,
            inputs=[obs_var, action_var] + state_info_vars_list + [valid_var],
            rnn_init_state=rnn_network.state_init_param,
            rnn_state_input=state_var,
            rnn_final_state=final_state,
        )

    def train(self):
        # use oracle policy to collect samples
        self.sampler.start_worker()
        self.eval_sampler.start_worker()
        paths = self.gen_training_data()
        self.init_opt()

        T = np.max([len(p["observations"]) for p in paths])
        observations = [p["observations"] for p in paths]
        observations = tensor_utils.pad_tensor_n(observations, T)
        actions = [p["actions"] for p in paths]
        actions = tensor_utils.pad_tensor_n(actions, T)
        valids = [np.ones((len(p["observations"], ))) for p in paths]
        valids = tensor_utils.pad_tensor_n(valids, T)

        n_train = int(np.floor(self.train_ratio * len(observations)))

        all_inputs = [observations, actions, valids]
        train_inputs = [x[:n_train] for x in all_inputs]
        test_inputs = [x[n_train:] for x in all_inputs]

        oracle_average_return = np.mean([np.sum(p["rewards"]) for p in paths])

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            assert len(self.policy.state_info_keys) == 0

            def itr_callback(itr, loss, learning_rate, best_loss, n_no_improvements, *args, **kwargs):
                params = self.policy.get_param_values(trainable=True)

                logger.log("Evaluating test loss")
                test_loss, _ = self.optimizer.loss_diagnostics(test_inputs)

                logger.log("Evaluating policy")
                eval_paths = self.eval_sampler.obtain_samples(
                    itr,
                    max_path_length=self.max_path_length,
                    batch_size=self.eval_batch_size
                )

                average_return = np.mean([np.sum(p["rewards"]) for p in eval_paths])

                logger.record_tabular("Epoch", itr)
                logger.record_tabular("TrainLoss", loss)
                logger.record_tabular("TestLoss", test_loss)
                logger.record_tabular("BestTrainLoss", best_loss)
                logger.record_tabular("NoImprovementEpochs", n_no_improvements)
                logger.record_tabular("LearningRate", learning_rate)
                logger.record_tabular('NPolicyParams', len(params))
                logger.record_tabular('AvgPolicyParamNorm', np.linalg.norm(params) / len(params))
                logger.record_tabular('OracleAverageReturn', oracle_average_return)
                logger.record_tabular('AverageReturn', average_return)
                logger.dump_tabular()
                return True

            self.optimizer.optimize(train_inputs, callback=itr_callback)

        self.sampler.shutdown_worker()
        self.eval_sampler.shutdown_worker()

