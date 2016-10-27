import os

import numpy as np

from rllab.misc.overrides import overrides

from sandbox.rein.algos.embedding_theano_par.batch_polopt import ParallelBatchPolopt
from sandbox.rein.algos.embedding_theano_par.conjugate_gradient_optimizer import \
    ParallelConjugateGradientOptimizer
import rllab.misc.logger as logger
from sandbox.rein.algos.embedding_theano_par.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.rein.algos.embedding_theano_par.plotter import Plotter

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
            clip_rewards=False,
            model_args=None,
            n_seq_frames=1,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ParallelConjugateGradientOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.mkl_num_threads = mkl_num_threads
        self._n_seq_frames = n_seq_frames

        assert model_args is not None
        assert eta >= 0
        assert train_model_freq >= 1
        if train_model:
            assert model_embedding

        self._eta = eta
        self._train_model = train_model
        self._train_model_freq = train_model_freq
        self._continuous_embedding = continuous_embedding
        self._model_embedding = model_embedding
        self._sim_hash_args = sim_hash_args
        self._clip_rewards = clip_rewards
        self._model_args = model_args

        if model_pool_args is None:
            self._model_pool_args = dict(size=100000, min_size=32, batch_size=32)
        else:
            self._model_pool_args = model_pool_args

        super(ParallelTRPOPlusLSH, self).__init__(**kwargs)

    def init_gpu(self):

        from sandbox.rein.algos.embedding_theano_par import parallel_trainer
        # start parallel trainer
        self._model_trainer = parallel_trainer.trainer
        logger.log('Sending model/pool data to parallel trainer and waiting for response ...')
        if self._train_model:
            # self._model_trainer.populate_trainer(self._model_args, self._model_pool_args)
            self._model_trainer.q_model_args.put(self._model_args)
            self._model_trainer.q_model_pool_args.put(self._model_pool_args)
            self._model_trainer.q_pool_data_out_flag.get()
        logger.log('Done.')

        from sandbox.rein.dynamics_models.bnn.conv_bnn_count import ConvBNNVIME

        self._model = ConvBNNVIME(
            **self._model_args
        )
        self._model_n_params = self._model.get_param_values().shape[0]

        if self._model_embedding:
            state_dim = self._model.discrete_emb_size
        else:
            state_dim = 128

        self._hashing_evaluator = ALEHashingBonusEvaluator(
            state_dim=state_dim,
            action_dim=0,
            count_target='embeddings',
            sim_hash_args=self._sim_hash_args,
            parallel=True
        )

        self._projection_matrix = np.random.normal(
            size=(self._model.discrete_emb_size, self._sim_hash_args['dim_key']))

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

        self._plotter = Plotter()

    @overrides
    def init_opt(self):
        """
        Same as normal NPO, except for setting MKL_NUM_THREADS.
        """
        import theano
        import theano.tensor as TT
        from rllab.misc import ext
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
        from rllab.misc import ext
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
            model=self._model
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
                # FIXME: to img change
                # obs_ed = self.decode_obs(self.encode_obs(path['env_infos']['images']))
                obs_ed = self.decode_obs(
                    self.encode_obs(path['observations'][:, -np.prod(self._model.state_dim):]))
                # Get continuous embedding.
                cont_emb = self._model.discrete_emb(obs_ed)
                if self._continuous_embedding:
                    return cont_emb
                else:
                    # Cast continuous embedding into binary one.
                    # return np.cast['int'](np.round(cont_emb))
                    bin_emb = np.cast['int'](np.round(cont_emb))
                    bin_emb_downsampled = bin_emb.reshape(-1, 8).mean(axis=1).reshape((bin_emb.shape[0], -1))
                    obs_key = np.cast['int'](np.round(bin_emb_downsampled))
                    obs_key[obs_key == 0] = -1
                    # obs_key = np.sign(np.asarray(obs_key).dot(self._projection_matrix))
                    return obs_key
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
                # self._hashing_evaluator_ram.fit_before_process_samples(paths)
                # for path in paths:
                #     path['ram_S'] = self._hashing_evaluator_ram.predict(path)
                # arr_surprise_ram = np.hstack([path['ram_S'] for path in paths])
                # logger.record_tabular('ram_MeanS', np.mean(arr_surprise_ram))
                # logger.record_tabular('ram_StdS', np.std(arr_surprise_ram))

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
            lst_obs_enc = []
            # Encode observations into replay pool format. Also make sure we only add final image in case of
            # autoencoder.
            # FIXME: to img change
            # obs_enc = self.encode_obs(path['env_infos']['images'][:, -np.prod(self._model.state_dim):])
            obs_enc = self.encode_obs(path['observations'][:, -np.prod(self._model.state_dim):])
            path_len = len(path['rewards'])
            tot_path_len += path_len
            for i in range(path_len):
                lst_obs_enc.append(obs_enc[i])
                # self._pool.add_sample(obs_enc[i])
            self._model_trainer.add_sample(lst_obs_enc)
        logger.log('{} samples added to replay pool'.format(tot_path_len))

    @overrides
    def encode_obs(self, obs):
        """
        Observation into uint8 encoding, also functions as target format
        """
        assert np.max(obs) <= 1.0
        assert np.min(obs) >= -1.0
        obs_enc = np.round((obs + 1.0) * 0.5 * (self._model.num_classes - 1)).astype("uint8")
        return obs_enc

    @overrides
    def decode_obs(self, obs):
        """
        From uint8 encoding to original observation format.
        """
        obs_dec = obs / float(self._model.num_classes - 1) * 2.0 - 1.0
        assert np.max(obs_dec) <= 1.0
        assert np.min(obs_dec) >= -1.0
        return obs_dec

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

        # # --
        # # Observations are concatenations of RAM and img.
        # # Actual observation = RAM, split off img into 'img'
        # for path in paths:
        #     path['images'] = np.array(path['observations'][128:])
        # --
        # Because the observations are received as single frames,
        # they have to be glued together again for n_seq_frames
        for path in paths:
            o = path['observations']
            o_ext = np.zeros((o.shape[0] + 3, o.shape[1]))
            o_ext[3:] = o
            o_lst = []
            for i in range(self._n_seq_frames - 1):
                o_lst.append(o_ext[i:i - self._n_seq_frames + 1])
            o_lst.append(o_ext[self._n_seq_frames - 1:])
            path['observations'] = np.concatenate(o_lst, axis=1)
