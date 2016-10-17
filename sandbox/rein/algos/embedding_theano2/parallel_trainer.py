from joblib.pool import MemmapingPool
import numpy as np
from sandbox.rein.algos.replay_pool import SingleStateReplayPool
from sandbox.rein.dynamics_models.utils import iterate_minibatches

import sys


class ParallelTrainer(object):
    def __init__(self):
        self._parallel_pool = MemmapingPool(
            1,
            temp_folder="/tmp",
            initializer=self.initialize()
        )
        self._model = None
        self._model_pool_args = None
        self._pool = None

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['_parallel_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def initialize(self):
        import theano.sandbox.cuda
        theano.sandbox.cuda.use("gpu")

    def terminate(self):
        self._parallel_pool.terminate()

    def add_sample(self, sample):
        self._pool.add_sample(sample)

    def populate_trainer(self, model=None, model_pool_args=None):
        assert model is not None
        assert model_pool_args is not None
        self._model = model
        self._model_pool_args = model_pool_args
        observation_dtype = "uint8"
        self._pool = SingleStateReplayPool(
            max_pool_size=model_pool_args['size'],
            observation_shape=(np.prod(self._model.state_dim),),
            observation_dtype=observation_dtype,
            **model_pool_args
        )

    def train_model(self):
        return self._parallel_pool.apply_async(self._train_model)

    def decode_obs(self, obs):
        """
        From uint8 encoding to original observation format.
        """
        obs_dec = obs / float(self._model.num_classes) * 2.0 - 1.0
        assert np.max(obs_dec) <= 1.0
        assert np.min(obs_dec) >= -1.0
        return obs_dec

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
            _o = self._model.pred_fn(_i)
            _o_s = _o.reshape((-1, np.prod(self._model.state_dim), self._model.num_classes))
            _o_s = np.argmax(_o_s, axis=2)
            acc += np.sum(np.abs(_o_s - _t))
        return acc / _inputs.shape[0]

    def _train_model(self):
        """
        Train autoencoder model.
        :return:
        """

        acc_before, acc_after, train_loss, running_avg = 0., 0., 0., 0.
        print('Updating autoencoder using replay pool ({}) ...'.format(self._pool.size))
        sys.stdout.flush()
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
            while done < 1:
                running_avg = 0.
                for _ in range(10):
                    # Replay pool return uint8 target format, so decode _x.
                    batch = self._pool.random_batch(self._model_pool_args['batch_size'])
                    _x = self.decode_obs(batch['observations'])
                    _y = batch['observations']
                    train_loss = float(self._model.train_fn(_x, _y, 0))
                    assert not np.isinf(train_loss)
                    assert not np.isnan(train_loss)
                    running_avg += train_loss / 100.
                running_avg_delta = old_running_avg - running_avg
                if running_avg_delta < 1e4:
                    done += 1
                else:
                    old_running_avg = running_avg
                    done = 0
                print('Autoencoder loss= {:.5f}, D= {:+.5f}, done={}'.format(
                    running_avg, running_avg_delta, done))
                sys.stdout.flush()

            for i in range(10):
                batch = self._pool.random_batch(32)
                _x = self.decode_obs(batch['observations'])
                _y = batch['observations']
                acc_after += self.accuracy(_x, _y) / 10.

            print('Autoencoder updated.')
            sys.stdout.flush()

        else:
            print('Autoencoder not updated: minimum replay pool size ({}) not met ({}).'.format(
                self._model_pool_args['min_size'], self._pool.size
            ))
            sys.stdout.flush()

        print("Done training.")
        sys.stdout.flush()
        return self._model.get_param_values()
