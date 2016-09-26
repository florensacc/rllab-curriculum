import tensorflow as tf
import time
from collections import defaultdict

from sandbox.rein.algos.embedding.plotter import Plotter
from sandbox.rein.algos.embedding.trpo import TRPO
from rllab.misc import special
import numpy as np
from rllab.misc import tensor_utils
import rllab.misc.logger as logger
from rllab.algos import util
from sandbox.rein.dynamics_models.utils import iterate_minibatches
from sandbox.rein.algos.replay_pool import SingleStateReplayPool

# --
# Nonscientific printing of numpy arrays.
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

CONSISTENCY_CHECK_DIR = '/consistency_check'
UNIQUENESS_CHECK_DIR = '/uniqueness_check'
RANDOM_SAMPLES_DIR = '/random_samples'


class TRPOPlus(TRPO):
    """
    TRPO+

    Extension to TRPO to allow for intrinsic reward.
    """

    def __init__(
            self,
            model=None,
            eta=0.1,
            model_pool_args=None,
            hamming_distance=0,
            **kwargs):
        super(TRPOPlus, self).__init__(**kwargs)

        assert model is not None

        self._model = model
        self._eta = eta
        self._hamming_distance = hamming_distance

        if model_pool_args is None:
            self._model_pool_args = dict(size=100000, min_size=32, batch_size=32)
        else:
            self._model_pool_args = model_pool_args

        observation_dtype = "uint8"
        self._pool = SingleStateReplayPool(
            max_pool_size=self._model_pool_args['size'],
            observation_shape=(self.env.observation_space.flat_dim,),
            observation_dtype=observation_dtype,
            **self._model_pool_args
        )
        self._plotter = Plotter(encode_obs=self.encode_obs, decode_obs=self.decode_obs,
                                normalize_obs=self.normalize_obs, denormalize_obs=self.denormalize_obs)

        # Counting table
        self._counting_table = defaultdict(lambda: 0)

    def process_samples(self, itr, paths):
        """
        Process samples.
        :param itr:
        :param paths:
        :return:
        """
        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + self.discount * path_baselines[1:] - path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.center_adv:
                advantages = util.center_advantages(advantages)

            if self.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["ext_rewards"]) for path in paths]

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["ext_rewards"]) for path in paths]

            ent = np.sum(self.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.log("Updating baseline ...")
        self.baseline.fit(paths)
        logger.log("Baseline updated.")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data

    def comp_int_rewards(self, paths):
        """
        Retrieve binary code and increase count of each sample in batch.
        :param paths:
        :return:
        """

        def bin_to_int(binary):
            integer = 0
            for bit in binary:
                integer = (integer << 1) | bit
            return integer

        def count_to_ir(count):
            return 1. / np.sqrt(count)

        for idx, path in enumerate(paths):
            # When using num_seq_frames > 1, we need to extract the last one.
            obs = self.decode_obs(path['observations'][:, -np.prod(self._model.state_dim):])
            keys = np.cast['int'](np.round(self._model.discrete_emb(obs)))
            counts = np.zeros(len(keys))
            lst_key_as_int = np.zeros(len(keys), dtype=int)
            for idy, key in enumerate(keys):
                key_as_int = bin_to_int(key)
                lst_key_as_int[idy] = key_as_int
                self._counting_table[key_as_int] += 1
                counts[idy] += self._counting_table[key_as_int]
                if self._hamming_distance == 1:
                    for i in range(len(key)):
                        key_trans = np.array(key)
                        key_trans[i] = 1 - key_trans[i]
                        key_trans_int = bin_to_int(key_trans)
                        # If you access the counting table directly, it puts a 0, which inflates the size.
                        if key_trans_int in self._counting_table.keys():
                            counts[idy] += self._counting_table[bin_to_int(key_trans)]

            path['S'] = count_to_ir(counts)

            num_unique = len(set(lst_key_as_int))
            logger.log('Path {}: unique values: {}/{}'.format(idx, num_unique, len(lst_key_as_int)))

        num_encountered_unique_keys = len(self._counting_table)
        logger.log('Counting table contains {} entries.'.format(num_encountered_unique_keys))

    def fill_replay_pool(self, paths):
        """
        Fill up replay pool.
        :param paths:
        :return:
        """
        tot_path_len = 0
        for path in paths:
            # Encode observations into replay pool format. Also make sure we only add final image in case of
            # autoencoder.
            obs_enc = self.encode_obs(path['observations'][:, -np.prod(self._model.state_dim):])
            path_len = len(path['rewards'])
            tot_path_len += path_len
            for i in range(path_len):
                self._pool.add_sample(obs_enc[i])
        logger.log('{} samples added to replay pool ({}).'.format(tot_path_len, self._pool.size))

    def encode_obs(self, obs):
        """
        Observation into uint8 encoding, also functions as target format
        """
        assert np.max(obs) <= 1.0
        return (obs * self._model.num_classes).astype("uint8")

    def decode_obs(self, obs):
        """
        From uint8 encoding to original observation format.
        """
        return obs / float(self._model.num_classes)

    def normalize_obs(self, obs):
        """
        Normalize observations.
        """
        shape = obs.shape
        o = obs.reshape((obs.shape[0], -1))
        mean, std = self._pool.get_cached_mean_std_obs()
        o = (o - mean[None, :]) / (std[None, :] + 1e-8)
        o = o.reshape(shape)
        return o

    def denormalize_obs(self, obs):
        """
        Denormalize observations.
        """
        shape = obs.shape
        o = obs.reshape((obs.shape[0], -1))
        mean, std = self._pool.get_cached_mean_std_obs()
        o = o * (std[None, :] + 1e-8) + mean[None, :]
        o = o.reshape(shape)
        return o

    def accuracy(self, _inputs, _targets):
        """
        Calculate accuracy for inputs/outputs.
        :param _inputs:
        :param _targets:
        :return:
        """
        acc = 0.
        for batch in iterate_minibatches(_inputs, _targets, 1000, shuffle=False):
            _i, _t, _ = batch
            # Decode observations.
            obs_dec_norm = self.normalize_obs(self.decode_obs(_i))
            _o = self._model.pred_fn(obs_dec_norm)
            _o_s = _o.reshape((-1, np.prod(self._model.state_dim), self._model.num_classes))
            _o_s = np.argmax(_o_s, axis=2)
            acc += np.sum(np.abs(_o_s - _t))
        return acc / _inputs.shape[0]

    def train_model(self):
        """
        Train autoencoder model.
        :return:
        """
        logger.log('Updating autoencoder using replay pool ({}) ...'.format(self._pool.size))
        acc_before, acc_after, train_loss, running_avg = 0., 0., 0., 0.
        if self._pool.size >= self._model_pool_args['min_size']:

            for _ in range(10):
                batch = self._pool.random_batch(32)
                _x = self.normalize_obs(self.decode_obs(batch['observations']))
                _y = batch['observations']
                acc_before += self.accuracy(_x, _y) / 10.

            done = False
            old_running_avg = np.inf
            while not done:
                running_avg = 0.
                for _ in range(50):
                    # Replay pool return uint8 target format, so decode _x.
                    batch = self._pool.random_batch(self._model_pool_args['batch_size'])
                    _x = self.normalize_obs(self.decode_obs(batch['observations']))
                    _y = batch['observations']
                    train_loss = float(self._model.train_fn(_x, _y, 0))
                    assert not np.isinf(train_loss)
                    assert not np.isnan(train_loss)
                    running_avg += train_loss / 100.
                if old_running_avg - running_avg < 1e4:
                    done = True
                logger.log('Autoencoder loss= {:.5f}\tD= {:.5f}'.format(
                    running_avg, old_running_avg - running_avg))
                old_running_avg = running_avg

            for i in range(10):
                batch = self._pool.random_batch(32)
                _x = self.normalize_obs(self.decode_obs(batch['observations']))
                _y = batch['observations']
                acc_after += self.accuracy(_x, _y) / 10.
            self._plotter.plot_pred_imgs(self._model, _x, _y, 0, 0, dir=RANDOM_SAMPLES_DIR)

            logger.log('Autoencoder updated.')
        else:
            logger.log('Autoencoder not updated: minimum replay pool size ({}) not met ({}).'.format(
                self._model_pool_args['min_size'], self._pool.size
            ))

        logger.record_tabular('AE_SqErrBefore', acc_before)
        logger.record_tabular('AE_SqErrAfter', acc_after)
        logger.record_tabular('AE_TrainLoss', running_avg)

    def add_int_to_ext_rewards(self, paths):
        """
        Alter rewards in-place.
        :param paths: sampled trajectories
        :return: None
        """
        for path in paths:
            path['rewards'] += self._eta * path['S']

    def preprocess(self, paths):
        """
        Preprocess data.
        :param paths:
        :return:
        """
        # --
        # Save external rewards.
        for path in paths:
            path['ext_rewards'] = np.array(path['rewards'])

    def diagnostics(self, start_time, itr, samples_data, paths):
        """
        Diagnostics of each run.
        :param start_time:
        :param itr:
        :param samples_data:
        :param paths:
        """
        # --
        # Analysis
        if itr == 0:
            # Select 5 random images form the first path, evaluate them at every iteration to inspect emb.
            rnd = np.random.randint(0, len(paths[0]['observations']), 5)
            self._test_obs = paths[0]['observations'][rnd]
        self._plotter.plot_pred_imgs(self._model, self.decode_obs(self._test_obs), self._test_obs, -itr - 1,
                                     0, dir=CONSISTENCY_CHECK_DIR)
        obs = paths[0]['observations'][-50:, -np.prod(self._model.state_dim):]
        self._plotter.plot_pred_imgs(self._model, (obs), obs, 0, 0, dir=UNIQUENESS_CHECK_DIR)

        # --
        # Diagnostics
        self.log_diagnostics(paths)
        logger.log("Saving snapshot ...")
        params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
        if self.store_paths:
            params["paths"] = samples_data["paths"]
        # FIXME: bugged: pickle issues
        # logger.save_itr_params(itr, params)
        logger.log("saved")
        logger.record_tabular('Time', time.time() - start_time)
        logger.dump_tabular(with_prefix=False)
        if self.plot:
            self.update_plot()
            if self.pause_for_plot:
                input("Plotting evaluation run: Press Enter to "
                      "continue...")

    def train(self):
        """
        Main RL training procedure.
        """
        # TODO: make sure normalize/decode/encode is applied correctly.
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                with logger.prefix('itr #%d | ' % itr):
                    # --
                    # Sample trajectories.
                    paths = self.obtain_samples(itr)

                    # --
                    # Preprocess trajectory data.
                    self.preprocess(paths)

                    # --
                    # Fill replay pool.
                    self.fill_replay_pool(paths)

                    # --
                    # Train model.
                    self.train_model()

                    # --
                    # Compute intrinisc rewards.
                    self.comp_int_rewards(paths)

                    # --
                    # Add intrinsic reward to external: 'rewards' is what is actually used as 'true' reward.
                    self.add_int_to_ext_rewards(paths)

                    # --
                    # Compute deltas, advantages, etc.
                    samples_data = self.process_samples(itr, paths)

                    # --
                    # Optimize policy according to latest trajectory batch `samples_data`.
                    self.optimize_policy(itr, samples_data)

                    # --
                    # Diagnosics
                    self.diagnostics(start_time, itr, samples_data, paths)

        self.shutdown_worker()
