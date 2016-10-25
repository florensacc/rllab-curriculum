import multiprocessing as mp
import numpy as np

import sys
import time
import sandbox.rein.algos.embedding_theano_par.n_parallel
import threading
import traceback

shared_model_params = mp.RawArray('d', 100000000)


class ParallelTrainer(object):
    def __init__(self):
        print(">>> Creating parallel trainer ...")
        self._parallel_pool = mp.Pool(
            1
        )
        self._n_parallel = sandbox.rein.algos.embedding_theano_par.n_parallel.n_parallel_
        self._model = None
        self._model_pool_args = None
        self._pool = None
        manager = mp.Manager()
        # Data q for input/output data
        self.q_pool_data_in = manager.Queue()
        self.q_pool_data_out = manager.Queue()
        self.q_pool_data_out_flag = manager.Queue()
        # Param q to input param if needed and to get the train_model exec running.
        self.q_train_flag = manager.Queue()
        # Output param q to return learned params.
        self.q_train_param_out = []
        self.q_train_acc_out = manager.Queue()
        for _ in range(self._n_parallel):
            self.q_train_param_out.append(manager.Queue())

        self.q_model_args = manager.Queue()
        self.q_model_pool_args = manager.Queue()
        print(">>> Parallel trainer created.")

    def _loop(self, q_pool_data_in=None, q_pool_data_out=None, q_pool_data_out_flag=None, q_train_flag=None,
              q_train_param_out=None, q_train_acc_out=None, q_model_args=None, q_model_pool_args=None):
        """
        Main Loop.
        :param data_q:
        :param param_q:
        :param output_q:
        :param model:
        :param model_pool_args:
        :return: None
        """
        try:
            print('>>> Start loop ...')
            sys.stdout.flush()
            # Init theano gpu context before any other theano context is initialized.
            import theano.sandbox.cuda
            theano.sandbox.cuda.use(
                "gpu")  # + str(7 - sandbox.rein.algos.embedding_theano_par.n_parallel._seed))
            # Init all main var + compile.
            from sandbox.rein.dynamics_models.bnn.conv_bnn_count import ConvBNNVIME
            print(">>> Theano imported.")
            sys.stdout.flush()
            sys.stderr.flush()

            model_args = None
            model_pool_args = None

            self.q_pool_data_out_flag.put(0)

            while model_args is None:
                if not q_model_args.empty():
                    model_args = q_model_args.get()

            print(">>> Model/pool data received.")
            sys.stdout.flush()

            model = ConvBNNVIME(
                **model_args
            )

            while model_pool_args is None:
                if not q_model_pool_args.empty():
                    model_pool_args = q_model_pool_args.get()

            model_pool_args = model_pool_args
            observation_dtype = "uint8"
            from sandbox.rein.algos.replay_pool import SingleStateReplayPool
            pool = SingleStateReplayPool(
                max_pool_size=model_pool_args['size'],
                observation_shape=(np.prod(model.state_dim),),
                observation_dtype=observation_dtype,
                **model_pool_args
            )
            print('>>> Compiled ({} params).'.format(len(model.get_param_values())))
            sys.stdout.flush()
            self.q_pool_data_out_flag.put(0)
            # Actual main loop.
            while True:
                while not q_pool_data_in.empty():
                    lst_sample = q_pool_data_in.get()
                    for sample in lst_sample:
                        pool.add_sample(sample)
                if not q_train_flag.empty():
                    q_train_flag.get()
                    self._train_model(q_train_param_out, q_train_acc_out, model, pool, model_pool_args)
                if not q_pool_data_out_flag.empty():
                    num_samples = q_pool_data_out_flag.get()
                    samples = pool.random_batch(num_samples)
                    q_pool_data_out.put(samples)
                time.sleep(0.1)
        except:
            # Log exceptions like this because mp is screwing me over.
            print("".join(traceback.format_exception(*sys.exc_info())))
            sys.stdout.flush()
            return None

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['_parallel_pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def terminate(self):
        self._parallel_pool.terminate()

    def add_sample(self, lst_sample):
        self.q_pool_data_in.put(lst_sample)

    def random_batch(self, num_samples):
        self.q_pool_data_out_flag.put(num_samples)
        return self.q_pool_data_out.get()

    def train_model(self):
        self.q_train_flag.put(0)

    def populate_trainer(self):
        print('Starting async model training loop.')

        self._parallel_pool.apply_async(
            self._loop,
            args=(self.q_pool_data_in, self.q_pool_data_out, self.q_pool_data_out_flag, self.q_train_flag,
                  self.q_train_param_out, self.q_train_acc_out, self.q_model_args, self.q_model_pool_args))

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
        from sandbox.rein.dynamics_models.utils import iterate_minibatches
        acc = 0.
        for batch in iterate_minibatches(_inputs, _targets, 1000, shuffle=False):
            _i, _t, _ = batch
            _o = model.pred_fn(_i)
            _o_s = _o.reshape((-1, np.prod(model.state_dim), model.num_classes))
            _o_s = np.argmax(_o_s, axis=2)
            acc += np.sum(np.abs(_o_s - _t))
        return acc / _inputs.shape[0]

    def _train_model(self, q_train_param_out=None, q_train_acc_out=None, model=None, pool=None,
                     model_pool_args=None):
        """
        Train autoencoder model.
        :return:
        """
        # import theano
        # def load_data():
        #     x_lst, y_lst = [], []
        #     for _ in range(itr_per_epoch):
        #         # Replay pool return uint8 target format, so decode _x.
        #         batch = pool.random_batch(model_pool_args['batch_size'])
        #         _x = self.decode_obs(batch['observations'], model)
        #         _y = batch['observations']
        #         x_lst.append(_x)
        #         y_lst.append(_y)
        #     x_arr = np.concatenate(x_lst, axis=0)
        #     y_arr = np.concatenate(y_lst, axis=0)
        #     model.shared_x.set_value(np.asarray(x_arr, dtype=theano.config.floatX))
        #     model.shared_y.set_value(np.asarray(y_arr, dtype='int32'))

        global shared_model_params
        assert q_train_param_out is not None
        assert model is not None
        assert pool is not None
        assert model_pool_args is not None
        itr_per_epoch = 100

        acc_before, acc_after, train_loss, running_avg = 0., 0., 0., 0.
        print('Updating autoencoder using replay pool ({}) ...'.format(pool.size))
        sys.stdout.flush()
        if pool.size >= model_pool_args['min_size']:

            # --
            # Actual training of model.
            done = 0
            old_running_avg = np.inf
            # first_run = True
            while done < 7:
                # Load first batch of data outside of thread.
                # load_data()
                running_avg = 0.
                start_time = time.time()
                # if not first_run:
                #     thread.join()
                # first_run = False
                # thread = threading.Thread(target=load_data)
                # thread.start()
                for i in range(itr_per_epoch):
                    # Replay pool return uint8 target format, so decode _x.
                    batch = pool.random_batch(model_pool_args['batch_size'])
                    _x = self.decode_obs(batch['observations'], model)
                    _y = batch['observations']
                    train_loss = float(model.train_fn(_x, _y, 0))
                    if np.isinf(train_loss):
                        # --
                        # Nonscientific printing of numpy arrays.
                        np.set_printoptions(suppress=True)
                        np.set_printoptions(precision=4)
                        np.set_printoptions(threshold=np.nan)
                        print(_x)
                        print(_y)
                        sys.stdout.flush()
                    assert not np.isinf(train_loss)
                    assert not np.isnan(train_loss)
                    running_avg += train_loss / float(itr_per_epoch)
                running_avg_delta = old_running_avg - running_avg
                if running_avg_delta < 1e-4:
                    done += 1
                else:
                    old_running_avg = running_avg
                    done = 0
                end_time = time.time()
                print('Autoencoder loss= {:.5f}, D= {:+.5f}, done={}\t{:.3f} sec/epoch'.format(
                    running_avg, running_avg_delta, done, (end_time - start_time) / float(itr_per_epoch)
                ))
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

        # Put updated params in the output queue.
        params = model.get_param_values()
        shared_model_params[:params.shape[0]] = params
        q_train_acc_out.put(acc_after)
        # Notify main processes that the params have been updated.
        for q in q_train_param_out:
            q.put(0)


trainer = ParallelTrainer()
