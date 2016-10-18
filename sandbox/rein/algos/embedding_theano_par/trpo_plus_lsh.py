import os

import theano
import theano.tensor as TT
import numpy as np

from rllab.misc import ext
from rllab.misc.overrides import overrides

from sandbox.rein.algos.embedding_theano_par.batch_polopt import ParallelBatchPolopt
from sandbox.rein.algos.embedding_theano_par.conjugate_gradient_optimizer import \
    ParallelConjugateGradientOptimizer
import rllab.misc.logger as logger
from sandbox.rein.algos.embedding_theano_par.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.rein.algos.embedding_theano_par.plotter import Plotter
from sandbox.rein.algos.embedding_theano_par.replay_pool import SingleStateReplayPool
from sandbox.rein.dynamics_models.utils import iterate_minibatches

# --
# Nonscientific printing of numpy arrays.
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

CONSISTENCY_CHECK_DIR = '/consistency_check'
UNIQUENESS_CHECK_DIR = '/uniqueness_check'
RANDOM_SAMPLES_DIR = '/random_samples'


class ParallelTRPOPlusLSH(ParallelBatchPolopt):
    """
    Parallelized Trust Region Policy Optimization (Synchronous)

    In this class definition, identical to serial case, except:
        - Inherits from parallelized base class
        - Holds a parallelized optimizer
        - Has an init_par_objs() method (working on base class and optimizer)
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            truncate_local_is_ratio=None,
            mkl_num_threads=1,
            eta=0.1,
            model_pool_args=None,
            train_model=True,
            train_model_freq=1,
            continuous_embedding=True,
            model_embedding=True,
            sim_hash_args=None,
            model=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ParallelConjugateGradientOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.mkl_num_threads = mkl_num_threads

        # # FIXME: this is a hack!
        # print('FIXME: this is a hack!! Only works for montezuma.')
        # import joblib
        # data = joblib.load('sandbox/rein/algos/embedding_theano_par/model/params.pkl')
        # model = data['model']

        assert eta >= 0
        assert train_model_freq >= 1
        if train_model:
            assert model_embedding
            assert model is not None

        self._model = model
        self._eta = eta
        self._train_model = train_model
        self._train_model_freq = train_model_freq
        self._continuous_embedding = continuous_embedding
        self._model_embedding = model_embedding
        self._sim_hash_args = sim_hash_args

        if self._model_embedding:
            state_dim = self._model.discrete_emb_size
            self._hashing_evaluator_ram = ALEHashingBonusEvaluator(
                state_dim=128,
                log_prefix="ram_",
                sim_hash_args=dict(
                    dim_key=256,
                    bucket_sizes=None,
                ),
                parallel=True
            )
        else:
            self._hashing_evaluator_ram = None
            state_dim = 128

        self._hashing_evaluator = ALEHashingBonusEvaluator(
            state_dim=state_dim,
            count_target='embeddings',
            sim_hash_args=sim_hash_args,
            parallel=True,
        )

        if self._model_embedding:
            logger.log('Model embedding enabled.')
            if self._train_model:
                logger.log('Training model enabled.')
            else:
                logger.log('Training model disabled, using convolutional random projection.')

            if self._continuous_embedding:
                logger.log('Using continuous embedding.')
            else:
                logger.log('Using binary embedding.')
        else:
            logger.log('Model embedding disabled, using LSH directly on states.')

        if model_pool_args is None:
            self._model_pool_args = dict(size=100000, min_size=32, batch_size=32)
        else:
            self._model_pool_args = model_pool_args

        if self._train_model:
            observation_dtype = "uint8"
            self._pool = SingleStateReplayPool(
                max_pool_size=self._model_pool_args['size'],
                observation_shape=(np.prod(self._model.state_dim),),
                observation_dtype=observation_dtype,
                **self._model_pool_args
            )
        self._plotter = Plotter()
        super(ParallelTRPOPlusLSH, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        """
        Same as normal NPO, except for setting MKL_NUM_THREADS.
        """
        # Set BEFORE Theano compiling; make equal to number of cores per worker.
        os.environ['MKL_NUM_THREADS'] = str(self.mkl_num_threads)

        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        input_list = [obs_var,
                      action_var,
                      advantage_var,
                      ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def prep_samples(self, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        return all_input_values

    @overrides
    def optimize_policy(self, itr, samples_data):
        if self.whole_paths:
            self.optimizer.set_avg_fac(self.n_steps_collected)  # (parallel)
        all_input_values = self.prep_samples(samples_data)
        self.optimizer.optimize(all_input_values)  # (parallel)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    @overrides
    def init_par_objs(self):
        self._init_par_objs_batchpolopt()  # (must do first)
        self.optimizer.init_par_objs(
            n_parallel=self.n_parallel,
            size_grad=len(self.policy.get_param_values(trainable=True)),
        )

    @overrides
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

        def obs_to_key(path):
            if self._model_embedding:
                # Encode/decode to get uniform representation.
                obs_ed = self.decode_obs(self.encode_obs(path['env_infos']['images']))
                # Get continuous embedding.
                cont_emb = self._model.discrete_emb(obs_ed)
                if self._continuous_embedding:
                    return cont_emb
                else:
                    # Cast continuous embedding into binary one.
                    return np.cast['int'](np.round(cont_emb))
            else:
                return path['observations']

        # --
        # Update counting table.
        if self.rank == 0:
            logger.log('Retrieve embeddings ...')
        for idx, path in enumerate(paths):
            # When using num_seq_frames > 1, we need to extract the last one.
            keys = obs_to_key(path)
            path['env_infos']['embeddings'] = keys

        if self.rank == 0:
            logger.log('Update counting table and compute intrinsic reward ...')
        self._hashing_evaluator.fit_before_process_samples(paths)
        for path in paths:
            path['S'] = self._hashing_evaluator.predict(path)

        arr_surprise = np.hstack([path['S'] for path in paths])
        logger.record_tabular('MeanS', np.mean(arr_surprise))
        logger.record_tabular('StdS', np.std(arr_surprise))

        if self._model_embedding:
            if self.rank == 0:
                logger.log('Update counting table and compute intrinsic reward (RAM) ...')
            self._hashing_evaluator_ram.fit_before_process_samples(paths)
            for path in paths:
                path['ram_S'] = self._hashing_evaluator_ram.predict(path)
            arr_surprise_ram = np.hstack([path['ram_S'] for path in paths])
            logger.record_tabular('ram_MeanS', np.mean(arr_surprise_ram))
            logger.record_tabular('ram_StdS', np.std(arr_surprise_ram))

        if self.rank == 0:
            logger.log('Intrinsic rewards computed')

    @overrides
    def fill_replay_pool(self, paths):
        """
        Fill up replay pool.
        :param paths:
        :return:
        """
        assert self._train_model
        tot_path_len = 0
        for path in paths:
            # Encode observations into replay pool format. Also make sure we only add final image in case of
            # autoencoder.
            obs_enc = self.encode_obs(path['env_infos']['images'][:, -np.prod(self._model.state_dim):])
            path_len = len(path['rewards'])
            tot_path_len += path_len
            for i in range(path_len):
                self._pool.add_sample(obs_enc[i])
        logger.log('{} samples added to replay pool ({}).'.format(tot_path_len, self._pool.size))

    @overrides
    def encode_obs(self, obs):
        """
        Observation into uint8 encoding, also functions as target format
        """
        assert np.max(obs) <= 1.0
        assert np.min(obs) >= -1.0
        obs_enc = np.round((obs + 1.0) * 0.5 * self._model.num_classes).astype("uint8")
        return obs_enc

    @overrides
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

    @overrides
    def train_model(self, itr):
        """
        Train autoencoder model.
        :return:
        """
        acc_before, acc_after, train_loss, running_avg = 0., 0., 0., 0.
        if itr == 0 or itr % self._train_model_freq == 0:
            logger.log('Updating autoencoder using replay pool ({}) ...'.format(self._pool.size))
            if self._pool.size >= self._model_pool_args['min_size']:

                for _ in range(100):
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
                self._plotter.print_embs(model=self._model, counting_table=None, inputs=_x,
                                         dir=RANDOM_SAMPLES_DIR, hamming_distance=0)

            else:
                logger.log('Autoencoder not updated: minimum replay pool size ({}) not met ({}).'.format(
                    self._model_pool_args['min_size'], self._pool.size
                ))

        logger.record_tabular('AE_SqErrBefore', acc_before)
        logger.record_tabular('AE_SqErrAfter', acc_after)
        logger.record_tabular('AE_TrainLoss', running_avg)

    @overrides
    def preprocess(self, paths):
        """
        Preprocess data.
        :param paths:
        :return:
        """
        # --
        # # Save external rewards.
        # for path in paths:
        #     path['raw_rewards'] = np.array(path['rewards'])

        # --
        # Observations are concatenations of RAM and img.
        # Actual observation = RAM, split off img into 'img'
        for path in paths:
            path['images'] = np.array(path['observations'][128:])
