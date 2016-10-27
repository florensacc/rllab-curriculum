import numpy as np

from sandbox.rein.algos.ganx_th.replay_pool import SingleStateReplayPool
from sandbox.rein.algos.ganx_th.plotter import Plotter
import rllab.misc.logger as logger

RANDOM_SAMPLES_DIR = '/random_samples'


class GANBonusEvaluator(object):
    """
    Using Generative Adverserial Nets that are trained frequently, and discriminator (D)
    as a means to detect novel samples. The deviation from a tracked D average is used
    as a novelty bonus.
    """

    def __init__(
            self,
            observation_shape=None,
            observation_dtype=None,
            train_model_freq=1,
            model_pool_args=None,
    ):

        assert observation_shape is not None
        assert observation_dtype is not None

        self._model = None
        self._train_model_freq = train_model_freq

        self._pool = SingleStateReplayPool(
            max_pool_size=model_pool_args['size'],
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            **model_pool_args
        )
        self._plotter = Plotter()

    def init_opt(self):
        """
        Here we compile everything.
        :return: None
        """
        self._model = None
        pass

    def train_model(self, itr):
        """
        Train model.
        :return: None
        """
        acc_before, acc_after, train_loss, running_avg = 0., 0., 0., 0.
        if itr == 0 or itr % self._train_model_freq == 0:
            logger.log('Updating autoencoder using replay pool ({}) ...'.format(self._pool.size))
            if self._pool.size >= self._model_pool_args['min_size']:

                for _ in range(10):
                    batch = self._pool.random_batch(32)
                    _x = self.decode_obs(batch['observations'])
                    _y = batch['observations']
                    acc_before += self.accuracy(_x, _y) / 10.

                # --
                # Actual training of model.
                done = 0
                old_running_avg = np.inf
                while done < 7:
                    running_avg = 0.
                    for _ in range(100):
                        # Replay pool return uint8 target format, so decode _x.
                        batch = self._pool.random_batch(self._model_pool_args['batch_size'])
                        _x = self.decode_obs(batch['observations'])
                        _y = batch['observations']
                        train_loss = float(self._model.train_fn(_x, _y, 0))
                        assert not np.isinf(train_loss)
                        assert not np.isnan(train_loss)
                        running_avg += train_loss / 100.
                    running_avg_delta = old_running_avg - running_avg
                    if running_avg_delta < 1e-4:
                        done += 1
                    else:
                        old_running_avg = running_avg
                        done = 0
                    logger.log('Autoencoder loss= {:.5f}, D= {:.5f}, done={}'.format(
                        running_avg, running_avg_delta, done))

                for i in range(10):
                    batch = self._pool.random_batch(32)
                    _x = self.decode_obs(batch['observations'])
                    _y = batch['observations']
                    acc_after += self.accuracy(_x, _y) / 10.

                logger.log('Autoencoder updated.')

                logger.log('Plotting random samples ...')
                self._plotter.plot_pred_imgs(model=self._model, inputs=_x, targets=_y, itr=0,
                                             dir=RANDOM_SAMPLES_DIR)
                self._plotter.print_embs(model=self._model, counting_table=self._counting_table, inputs=_x,
                                         dir=RANDOM_SAMPLES_DIR, hamming_distance=self._hamming_distance)

            else:
                logger.log('Autoencoder not updated: minimum replay pool size ({}) not met ({}).'.format(
                    self._model_pool_args['min_size'], self._pool.size
                ))

        logger.record_tabular('AE_SqErrBefore', acc_before)
        logger.record_tabular('AE_SqErrAfter', acc_after)
        logger.record_tabular('AE_TrainLoss', running_avg)

    def encode_obs(self, obs):
        """
        Observation into uint8 encoding, also functions as target format
        """
        assert np.max(obs) <= 1.0
        assert np.min(obs) >= -1.0
        obs_enc = np.round((obs + 1.0) * 0.5 * self._model.num_classes).astype("uint8")
        return obs_enc

    def decode_obs(self, obs):
        """
        From uint8 encoding to original observation format.
        """
        obs_dec = obs / float(self._model.num_classes) * 2.0 - 1.0
        assert np.max(obs_dec) <= 1.0
        assert np.min(obs_dec) >= -1.0
        return obs_dec

    def preprocess(self, states):
        """
        Preprocess, also feed into replay pool.
        :param states: that on which the model acts.
        :return: None
        """

        pass

    def before_process_samples(self, paths):
        # --
        # Fill replay pool
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

    def compute_bonus(self, path):
        """
        Apply discriminator (D) to all states in path.
        :param path: sampled trajectory
        :return: None
        """
        pass

    def after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, paths):
        """
        Sample images using Plotter.
        :param paths:
        :return: None
        """
        pass
