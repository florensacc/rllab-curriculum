from joblib.pool import MemmapingPool
import multiprocessing as mp
import numpy as np
from sandbox.rein.algos.replay_pool import SingleStateReplayPool
from sandbox.rein.dynamics_models.utils import iterate_minibatches

import sys
import time
import pickle


class ParallelTrainer(object):
    def __init__(self):
        self._parallel_pool = MemmapingPool(
            1,
            temp_folder="/tmp",
            initializer=self._initialize
        )
        self._model = None
        self._model_pool_args = None
        self._pool = None
        manager = mp.Manager()
        # Data q for input data
        self.data_q = manager.Queue()
        # Param q to input param if needed and to get the train_model exec running.
        self.param_q = manager.Queue()
        # Output param q to return learned params.
        self.output_q = manager.Queue()

    @staticmethod
    def _initialize():
        # Init theano gpu context before any other theano context is initialized.
        import theano.sandbox.cuda
        theano.sandbox.cuda.use("gpu")

    def _loop(self, data_q=None, param_q=None, output_q=None, model=None, model_pool_args=None):
        """
        Main Loop.
        :param data_q:
        :param param_q:
        :param output_q:
        :param model:
        :param model_pool_args:
        :return: None
        """
        assert model is not None
        assert model_pool_args is not None
        # Init all main var + compile.
        model = pickle.loads(model)
        model_pool_args = model_pool_args
        observation_dtype = "uint8"
        pool = SingleStateReplayPool(
            max_pool_size=model_pool_args['size'],
            observation_shape=(np.prod(model.state_dim),),
            observation_dtype=observation_dtype,
            **model_pool_args
        )
        # Actual main loop.
        while True:
            while not data_q.empty():
                pool.add_sample(data_q.get())
            if not param_q.empty():
                param_q.get()
                self._train_model(output_q, model, pool, model_pool_args)
            time.sleep(0.1)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['_parallel_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def terminate(self):
        self._parallel_pool.terminate()

    def add_sample(self, sample):
        self.data_q.put(sample)

    def train_model(self):
        self.param_q.put(0)

    def populate_trainer(self, model=None, model_pool_args=None):
        print('Starting async model training loop.')
        self._parallel_pool.apply_async(self._loop,
                                        args=(self.data_q, self.param_q, self.output_q, model, model_pool_args))

    @staticmethod
    def decode_obs(obs, model):
        """
        From uint8 encoding to original observation format.
        """
        obs_dec = obs / float(model.num_classes) * 2.0 - 1.0
        assert np.max(obs_dec) <= 1.0
        assert np.min(obs_dec) >= -1.0
        return obs_dec

    @staticmethod
    def accuracy(_inputs, _targets, model):
        """
        Calculate accuracy for inputs/outputs.
        :param _inputs:
        :param _targets:
        :return:
        """
        acc = 0.
        for batch in iterate_minibatches(_inputs, _targets, 1000, shuffle=False):
            _i, _t, _ = batch
            _o = model.pred_fn(_i)
            _o_s = _o.reshape((-1, np.prod(model.state_dim), model.num_classes))
            _o_s = np.argmax(_o_s, axis=2)
            acc += np.sum(np.abs(_o_s - _t))
        return acc / _inputs.shape[0]

    def _train_model(self, output_q=None, model=None, pool=None, model_pool_args=None):
        """
        Train autoencoder model.
        :return:
        """
        assert output_q is not None
        assert model is not None
        assert pool is not None
        assert model_pool_args is not None

        acc_before, acc_after, train_loss, running_avg = 0., 0., 0., 0.
        print('Updating autoencoder using replay pool ({}) ...'.format(pool.size))
        sys.stdout.flush()
        if pool.size >= model_pool_args['min_size']:

            for _ in range(10):
                batch = pool.random_batch(32)
                _x = self.decode_obs(batch['observations'], model)
                _y = batch['observations']
                acc_before += self.accuracy(_x, _y, model) / 10.

            # --
            # Actual training of model.
            done = 0
            old_running_avg = np.inf
            while done < 1:
                running_avg = 0.
                for _ in range(10):
                    # Replay pool return uint8 target format, so decode _x.
                    batch = pool.random_batch(model_pool_args['batch_size'])
                    _x = self.decode_obs(batch['observations'], model)
                    _y = batch['observations']
                    train_loss = float(model.train_fn(_x, _y, 0))
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
                batch = pool.random_batch(32)
                _x = self.decode_obs(batch['observations'], model)
                _y = batch['observations']
                acc_after += self.accuracy(_x, _y, model) / 10.

            print('Autoencoder updated.')
            sys.stdout.flush()

        else:
            print('Autoencoder not updated: minimum replay pool size ({}) not met ({}).'.format(
                model_pool_args['min_size'], pool.size
            ))
            sys.stdout.flush()

        print("Done training.")
        sys.stdout.flush()

        # Put updated params in the output queue.
        output_q.put(model.get_param_values())
