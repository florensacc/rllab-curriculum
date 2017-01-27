import tensorflow as tf

from rllab.algos import util
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import special
from rllab.misc.instrument import VariantGenerator, variant
from sandbox.rocky.s3.resource_manager import resource_manager
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
import cloudpickle

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.distributions.bernoulli import Bernoulli
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.sample_processors.default_sample_processor import DefaultSampleProcessor
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP

from rllab.misc import logger

import numpy as np


class GAILSampleProcessor(DefaultSampleProcessor):
    def __init__(self, algo, discriminator):
        DefaultSampleProcessor.__init__(self, algo)
        self.discriminator = discriminator

    def process_samples(self, itr, paths):
        self.discriminator.fit(paths)
        for path, rewards in zip(paths, self.discriminator.batch_predict(paths)):
            path["raw_rewards"] = path["rewards"]
            path["rewards"] = rewards

        baselines = []
        returns = []

        if len(paths) > 0 and "vf" in paths[0]["agent_infos"]:
            all_path_baselines = [
                p["agent_infos"]["vf"].flatten() for p in paths
                ]
        else:
            if hasattr(self.algo.baseline, "predict_n"):
                all_path_baselines = self.algo.baseline.predict_n(paths)
            else:
                all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            path["raw_returns"] = special.discount_cumsum(path["raw_rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [np.sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [np.sum(path["rewards"]) for path in paths]

            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths)
        logger.log("fitted")

        average_discounted_raw_return = \
            np.mean([path["raw_returns"][0] for path in paths])
        undiscounted_raw_returns = [np.sum(path["raw_rewards"]) for path in paths]

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageDiscountedRawReturn',
                              average_discounted_raw_return)
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular_misc_stat('TrajLen', [len(p["rewards"]) for p in paths], placement='front')
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular_misc_stat('Return', undiscounted_returns, placement='front')
        logger.record_tabular_misc_stat('RawReturn', undiscounted_raw_returns, placement='front')

        return samples_data


class MLPDiscriminator(object):
    def __init__(self, env_spec, demo_paths, n_epochs=2, batch_size=128, learning_rate=1e-3):
        self.env_spec = env_spec
        self.demo_paths = demo_paths

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        l_obs = L.InputLayer(
            shape=(None, obs_dim),
        )

        l_action = L.InputLayer(
            shape=(None, action_dim),
        )

        network = MLP(
            name="disc",
            input_shape=(obs_dim + action_dim,),
            input_layer=L.concat([l_obs, l_action], axis=1),
            output_dim=1,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
        )

        self.network = network
        self.f_predict = tensor_utils.compile_function(
            inputs=[l_obs.input_var, l_action.input_var],
            outputs=L.get_output(network.output_layer),
        )
        self.l_obs = l_obs
        self.l_action = l_action
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_opt()

    def init_opt(self):
        obs_var = self.l_obs.input_var
        action_var = self.l_action.input_var
        y_var = tf.placeholder(dtype=tf.float32, shape=(None,), name="y")
        logits = L.get_output(self.network.output_layer)[:, 0]
        predict_p = tf.sigmoid(logits)

        cross_ent_loss = tf.reduce_mean(y_var * -tf.log(predict_p + 1e-8) + (1 - y_var) * -tf.log(1 - predict_p + 1e-8))
        ent = tf.reduce_mean(predict_p * -tf.log(predict_p + 1e-8) + (1 - predict_p) * -tf.log(1 - predict_p + 1e-8))

        loss = cross_ent_loss - 0.01 * ent

        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            loss, var_list=self.network.get_params(trainable=True))

        self.f_train = tensor_utils.compile_function(
            inputs=[obs_var, action_var, y_var],
            outputs=train_op,
        )
        self.f_loss = tensor_utils.compile_function(
            inputs=[obs_var, action_var, y_var],
            outputs=loss,
        )
        self.f_predict = tensor_utils.compile_function(
            inputs=[obs_var, action_var],
            outputs=predict_p,
        )

    def fit(self, paths):
        # Refit the discriminator on the new data
        demo_observations = tensor_utils.concat_tensor_list([p["observations"] for p in self.demo_paths])
        demo_actions = tensor_utils.concat_tensor_list([p["actions"] for p in self.demo_paths])
        demo_labels = np.ones((len(demo_observations),))
        demo_data = [demo_observations, demo_actions, demo_labels]
        demo_N = len(demo_observations)

        pol_observations = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        pol_actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        pol_labels = np.zeros((len(pol_observations),))
        pol_data = [pol_observations, pol_actions, pol_labels]
        pol_N = len(pol_observations)

        joint_data = [np.concatenate([x, y], axis=0) for x, y in zip(demo_data, pol_data)]

        loss_before = self.f_loss(*joint_data)

        # one epoch is as large as the minimum size of policy / demo paths
        for _ in range(self.n_epochs):
            # shuffling all data
            demo_ids = np.arange(demo_N)
            np.random.shuffle(demo_ids)
            demo_data = [x[demo_ids] for x in demo_data]
            pol_ids = np.arange(pol_N)
            np.random.shuffle(pol_ids)
            pol_data = [x[pol_ids] for x in pol_data]

            for batch_idx in range(0, min(demo_N, pol_N), self.batch_size):
                # take samples from each sides
                demo_batch = [x[batch_idx:batch_idx + self.batch_size] for x in demo_data]
                pol_batch = [x[batch_idx:batch_idx + self.batch_size] for x in pol_data]
                joint_batch = [np.concatenate([x, y], axis=0) for x, y in zip(demo_batch, pol_batch)]
                self.f_train(*joint_batch)

        loss_after = self.f_loss(*joint_data)

        logger.record_tabular('DiscLossBefore', loss_before)
        logger.record_tabular('DiscLossAfter', loss_after)

    def batch_predict(self, paths):
        pol_observations = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        pol_actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        ps = self.f_predict(pol_observations, pol_actions)
        start_idx = 0
        ret = []
        for path in paths:
            ret.append(ps[start_idx:start_idx + len(path["rewards"])])
            start_idx += len(path["rewards"])
        return ret

class VG(VariantGenerator):
    @variant
    def task(self):
        return ["cartpole", "cartpole_swing_up", "double_pendulum", "mountain_car"]
        # return TASKS.keys()

def main():
    with tf.Session() as sess:
        # traj_resource_name = "demo_trajs/cartpole_1000.pkl"

        task = "cartpole"
        n_trajs = 1000
        horizon = 500
        deterministic = True

        traj_resource_name = "demo_trajs/{task}_n_trajs_{n_trajs}_horizon_{horizon}_deterministic_{" \
                             "deterministic}.pkl".format(task=task, n_trajs=str(n_trajs), horizon=str(
            horizon), deterministic=str(deterministic))

        with open(resource_manager.get_file(traj_resource_name), "rb") as f:
            demo_paths = cloudpickle.load(f)

        env = TfEnv(normalize(CartpoleEnv()))

        discriminator = MLPDiscriminator(env_spec=env.spec, demo_paths=demo_paths)

        policy = GaussianMLPPolicy(name="policy", env_spec=env.spec)
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=50000,
            max_path_length=500,
            sample_processor_cls=GAILSampleProcessor,
            sample_processor_args=dict(discriminator=discriminator),
        )

        algo.train(sess=sess)


if __name__ == "__main__":
    main()
