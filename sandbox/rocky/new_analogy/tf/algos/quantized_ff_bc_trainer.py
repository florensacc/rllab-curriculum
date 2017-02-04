from rllab.core.serializable import Serializable
import tensorflow as tf

from rllab.envs.base import Env
from rllab.envs.env_spec import EnvSpec
from sandbox.rocky.tf.core.parameterized import JointParameterized
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import numpy as np
from rllab.misc import logger
from sandbox.rocky.tf.spaces import Discrete


class Preprocessor(object):
    def __init__(self, data):
        from matplotlib.mlab import PCA
        pca = PCA(data)
        self.pca = pca

    def whiten_n(self, data):
        data = (data - self.pca.mu) / (self.pca.sigma)
        return data.dot(self.pca.Wt.T)

    def unwhiten_n(self, data):
        return data.dot(self.pca.Wt) * self.pca.sigma + self.pca.mu


class NoopPreprocessor(object):
    def whiten_n(self, data):
        return data

    def unwhiten_n(self, data):
        return data


class PreprocessedPolicy(object):
    def __init__(self, wrapped_policy, quantizer):  # preprocess_obs, preprocess_actions):
        self.wrapped_policy = wrapped_policy
        self.quantizer = quantizer
        # self.preprocess_obs = preprocess_obs
        # self.preprocess_actions = preprocess_actions

    def get_action(self, obs):
        actions, agent_infos = self.get_actions([obs])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        obs_space = self.wrapped_policy.observation_space
        # action_space = self.wrapped_policy.action_space
        # flat_obs = obs_space.flatten_n(observations)
        # flat_whitened_obs = self.preprocess_obs.whiten_n(flat_obs)
        # whitened_obs = obs_space.unflatten_n(flat_whitened_obs)
        actions, agent_infos = self.wrapped_policy.get_actions(observations)
        actions = self.quantizer.unquantize(obs_space.flatten_n(observations), actions)
        # whitened_action = action_space.flatten_n(actions)
        # actions = action_space.unflatten_n(self.preprocess_actions.unwhiten_n(whitened_action))
        return actions, agent_infos

    def reset(self):
        self.wrapped_policy.reset()


class ComponentPolicy(object):
    def __init__(self, policy_list):
        self.observation_space = policy_list[0].observation_space
        self.policy_list = policy_list

    def get_actions(self, observations):
        actions = []
        for policy in self.policy_list:
            actions.append(policy.get_actions(observations)[0])
        return np.asarray(actions), dict()

    def reset(self):
        for policy in self.policy_list:
            policy.reset()


class ActionQuantizer(object):
    def __init__(self, observations, actions, n_bins=100):
        self.observations = np.asarray(observations)
        self.actions = np.asarray(actions)

        self.action_dim = action_dim = actions.shape[-1]
        diff_actions = actions - observations[:, :action_dim]

        action_quants = []

        for action_idx in range(action_dim):
            quants = []
            ith_actions = np.sort(diff_actions[:, action_idx])

            for quant_idx in range(n_bins):
                median = ith_actions[int((quant_idx + 0.5) / n_bins * len(actions))]
                quants.append(median)

            action_quants.append(quants)

        self.action_quants = np.asarray(action_quants)

    def quantize(self, observations, actions):
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        diff_actions = actions - observations[:, :self.action_dim]
        return np.argmin(np.abs(diff_actions[:, :, np.newaxis] - self.action_quants[np.newaxis, :, :]), axis=2)

    def unquantize(self, observations, quantized_actions):
        observations = np.asarray(observations)
        quantized_actions = np.asarray(quantized_actions)
        actions = self.action_quants[
            np.repeat(np.arange(self.action_dim), len(observations)),
            quantized_actions.flatten()
        ].reshape((len(observations), -1))
        return actions + observations[:, :self.action_dim]


class Trainer(Serializable):
    def __init__(
            self,
            env: Env,
            # policy,
            paths,
            threshold=None,
            n_quant_actions=50,
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
        if threshold is not None:
            paths = [p for p in paths if p["rewards"][-1] >= threshold]

        n_train = int(train_ratio * len(paths))

        np.random.shuffle(paths)

        train_paths = paths[:n_train]
        test_paths = paths[n_train:]
        self.n_quant_actions = n_quant_actions
        self.quant_action_space = Discrete(n_quant_actions)
        self.policy_list = []
        for idx in range(env.action_dim):
            self.policy_list.append(
                CategoricalMLPPolicy(
                    hidden_sizes=(128, 128),
                    env_spec=EnvSpec(
                        observation_space=env.observation_space,
                        action_space=self.quant_action_space,
                    ),
                    hidden_nonlinearity=tf.nn.relu,
                    name="policy_%d" % idx
                )
            )
        self.policy = ComponentPolicy(self.policy_list)#policy
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
        action_var = self.quant_action_space.new_tensor_variable(name="action", extra_dims=2)

        loss = 0
        for idx, policy in enumerate(self.policy_list):
            dist_info = policy.dist_info_sym(obs_var=obs_var)
            # minimize the M-projection KL divergence
            loss += -tf.reduce_mean(policy.distribution.log_likelihood_sym(action_var[:, idx, :], dist_info))
        loss /= len(self.policy_list)

        self.optimizer = FirstOrderOptimizer(
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=dict(learning_rate=self.learning_rate),
            max_epochs=self.n_epochs,
            batch_size=self.batch_size,
            tolerance=None,
            verbose=True,
        )

        self.optimizer.update_opt(loss=loss, target=JointParameterized(self.policy_list), inputs=[obs_var, action_var])

    def train(self, sess=None):
        self.init_opt()

        session_created = False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
            session_created = True
        # Only initialize variables that have not been initialized

        tensor_utils.initialize_new_variables(sess=sess)

        train_observations = np.concatenate([p["observations"] for p in self.train_paths])
        train_actions = np.concatenate([p["actions"] for p in self.train_paths])

        quantizer = ActionQuantizer(train_observations, train_actions, n_bins=self.n_quant_actions)

        # whiten_obs = Preprocessor(train_observations)
        # whiten_actions = NoopPreprocessor()#Preprocessor(train_actions)
        # train_observations = whiten_obs.whiten_n(train_observations)
        # import ipdb;
        # ipdb.set_trace()
        train_actions = self.quant_action_space.flatten_n(
            quantizer.quantize(train_observations, train_actions).flatten()
        ).reshape((len(train_observations), self.env.action_dim, -1))  # whiten_actions.whiten_n(
        # train_actions)

        # import ipdb; ipdb.set_trace()

        # all_actions = np.fromfile(open(file_name, 'rb')).reshape((100, 1000, 8))

        # import matplotlib.pyplot as plt
        # for idx in range(8):
        #     plt.subplot(2, 4, idx + 1)
        #     plt.hist(train_actions[:, idx].flatten(), 10)
        # plt.show()

        if len(self.test_paths) > 0:
            test_observations = np.concatenate([p["observations"] for p in self.test_paths])
            test_actions = np.concatenate([p["actions"] for p in self.test_paths])
            # test_observations = whiten_obs.whiten_n(test_observations)
            test_actions = self.quant_action_space.flatten_n(
                quantizer.quantize(test_observations, test_actions).flatten()
            ).reshape((len(test_observations), self.env.action_dim, -1))

        eval_policy = PreprocessedPolicy(self.policy, quantizer)  # whiten_obs, whiten_actions)

        if self.evaluate_performance:
            sampler = VectorizedSampler(
                env=self.env,
                policy=eval_policy,
                n_envs=self.n_eval_envs
            )
            logger.log("Starting worker...")
            sampler.start_worker()
            logger.log("Worker started")
            # sess.run(tf.assign(self.policy._l_std_param.param, [-10] * self.env.action_dim))

            # test_actions)

        # import ipdb; ipdb.set_trace()

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

            logger.save_itr_params(itr, dict(env=self.env, policy=eval_policy))

        self.optimizer.optimize(inputs=[train_observations, train_actions], callback=cb)

        if self.evaluate_performance:
            sampler.shutdown_worker()
        if session_created:
            sess.__exit__(None, None, None)
