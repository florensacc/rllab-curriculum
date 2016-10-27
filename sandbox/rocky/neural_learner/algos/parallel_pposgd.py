from collections import OrderedDict

import tensorflow as tf
import numpy as np
from cached_property import cached_property

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc import logger as root_logger
from rllab.misc import ext
from sandbox.rocky.analogy.rnn_cells import GRUCell
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
import sandbox.rocky.analogy.core.layers as LL
from sandbox.rocky.tf.core.parameterized import suppress_params_loading, Parameterized
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.distributions.recurrent_diagonal_gaussian import RecurrentDiagonalGaussian
from sandbox.rocky.tf.misc import tensor_utils

# import tensorflow as tf, rl_algs.common.tf_util as U, numpy as np
from sandbox.rocky.tf.spaces import Discrete, Box
import os
import subprocess
import sys
from mpi4py import MPI

import builtins
if not hasattr(builtins, 'profile'):

    def profile(fn, *args, **kwargs):
        return fn

    builtins.profile = profile



class MpiLogger(object):
    def __init__(self):
        self.entries = []
        pass

    def record_tabular(self, key, value, op):
        self.entries.append((key, value, op))

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" + suffix, np.average(values), op=np.nanmean)
            self.record_tabular(prefix + "Std" + suffix, np.std(values), op=np.nanmean)
            self.record_tabular(prefix + "Median" + suffix, np.median(values), op=np.nanmean)
            self.record_tabular(prefix + "Min" + suffix, np.min(values), op=np.nanmin)
            self.record_tabular(prefix + "Max" + suffix, np.max(values), op=np.nanmax)
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan, op=np.nanmean)
            self.record_tabular(prefix + "Std" + suffix, np.nan, op=np.nanmean)
            self.record_tabular(prefix + "Median" + suffix, np.nan, op=np.nanmean)
            self.record_tabular(prefix + "Min" + suffix, np.nan, op=np.nanmin)
            self.record_tabular(prefix + "Max" + suffix, np.nan, op=np.nanmax)

    def dump_tabular(self):
        data = MPI.COMM_WORLD.gather(self.entries, root=0)
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            for entries in zip(*data):
                keys, values, ops = zip(*entries)
                key = keys[0]
                value = ops[0](values)
                root_logger.record_tabular(key, value)
            root_logger.dump_tabular()
        self.entries = []


class MpiAdam(object):
    def __init__(self, var_list, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08, sync_frequency=100):
        self.var_list = var_list
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        size = sum(tensor_utils.flat_dim(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0

        flat_var = tf.placeholder(dtype=tf.float32, shape=(size,), name="flat")
        shapes = tensor_utils.tensor_shapes(var_list)

        self.getflat = tensor_utils.compile_function(
            inputs=var_list,
            outputs=tensor_utils.flatten_tensor_variables(var_list)
        )
        self.setfromflat = tensor_utils.compile_function(
            inputs=[flat_var],
            outputs=[tf.assign(v, v_new) for v, v_new in zip(
                var_list,
                tensor_utils.unflatten_tensor_variables(flat_var, shapes=shapes, symb_arrs=var_list)
            )]
        )
        self.sync_frequency = sync_frequency
        # self.setfromflat = U.SetFromFlat(var_list)
        # self.getflat = U.GetFlat(var_list)

    def update(self, localg):
        if self.t % self.sync_frequency == 0:
            self.sync()

        globalg = np.zeros_like(localg)
        MPI.COMM_WORLD.Allreduce(localg, globalg, op=MPI.SUM)
        globalg /= MPI.COMM_WORLD.Get_size()

        self.t += 1
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = - a * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        """
        Set everyone's parameters to worker 0's parameters
        """
        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            theta = self.getflat()
            MPI.COMM_WORLD.Bcast(theta, root=0)
        else:
            theta = np.empty_like(self.m)
            MPI.COMM_WORLD.Bcast(theta, root=0)
            self.setfromflat(theta)


# class RunningMeanStd(object):
#     """
#     Taken from John's code
#     """
#
#     # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#     def __init__(self, shape=(), momentum=0.9):
#         self.mean = old_mean = tf.get_variable(name="runningmean",
#                                                shape=shape,
#                                                dtype=tf.float32,
#                                                initializer=tf.zeros_initializer,
#                                                trainable=False)
#
#         self.var = old_var = tf.get_variable(name="runningvar",
#                                              shape=shape,
#                                              dtype=tf.float32,
#                                              initializer=tf.ones_initializer,
#                                              trainable=False)
#
#         self.count = old_count = tf.get_variable(name="count",
#                                                  shape=(),
#                                                  dtype=tf.float32,
#                                                  initializer=tf.constant_initializer(epsilon),
#                                                  trainable=False)
#
#         self.std = tf.sqrt(self.var)
#
#         self.x = batch_x = tf.placeholder(dtype=tf.float32, shape=[None] + list(shape))
#         batch_mean, batch_var = tf.nn.moments(batch_x, axes=[0])
#         batch_count = tf.to_float(tf.shape(batch_x)[0])
#
#         delta = batch_mean - old_mean
#         tot_count = old_count + batch_count
#
#         new_mean = old_mean * momentum + batch_mean * (1 - momentum)# + + delta * batch_count / tot_count
#
#         # m_a = old_var * (old_count)
#         # m_b = batch_var * (batch_count)
#         # M2 = m_a + m_b + tf.square(delta) * old_count * batch_count / (old_count + batch_count)
#         new_var = old_var * momentum + batch_var * (1 - momentum)#M2 / (old_count + batch_count)
#
#         # new_count = batch_count + old_count
#
#         with tf.control_dependencies([new_mean, new_var]):
#             self._updates = [
#                 tf.assign(old_mean, new_mean),
#                 tf.assign(old_var, new_var),
#                 # tf.assign(old_count, new_count)
#             ]
#
#     def update(self, x):
#         # pass
#         tf.get_default_session().run(self._updates, feed_dict={self.x: x})
#
#     @property
#     def variables(self):
#         return [self.mean, self.var]#, self.count]


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):
        self._sum = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(0.0),
            name="runningsum", trainable=False)
        self._sumsq = tf.get_variable(
            dtype=tf.float64,
            shape=shape,
            initializer=tf.constant_initializer(epsilon),
            name="runningsumsq", trainable=False)
        self._count = tf.get_variable(
            dtype=tf.float64,
            shape=(),
            initializer=tf.constant_initializer(epsilon),
            name="count", trainable=False)
        self.shape = shape

        self.mean = tf.to_float(self._sum / self._count)
        self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2))

        newsum = tf.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        newsumsq = tf.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        newcount = tf.placeholder(shape=[], dtype=tf.float64, name='count')
        self.incfiltparams = tensor_utils.compile_function(
            inputs=[newsum, newsumsq, newcount],
            outputs=[tf.assign_add(self._sum, newsum),
                     tf.assign_add(self._sumsq, newsumsq),
                     tf.assign_add(self._count, newcount[0])]
        )
        # ).function([newsum, newsumsq, newcount], [],
        #     updates=[tf.assign_add(self._sum, newsum),
        #              tf.assign_add(self._sumsq, newsumsq),
        #              tf.assign_add(self._count, newcount[0])])

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n * 2 + 1, 'float64')
        addvec = np.concatenate(
            [x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)], dtype='float64')])
        MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(totalvec[0:n].reshape(self.shape), totalvec[n:2 * n].reshape(self.shape), totalvec[2 * n])


class RNNActorCritic(LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_dim=256,
            hidden_nonlinearity=tf.nn.relu,
    ):
        Serializable.quick_init(self, locals())
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        with tf.variable_scope(name):
            self.observation_space = env_spec.observation_space
            self.action_space = env_spec.action_space

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            l_obs_inputs = self.new_obs_input_layers(extra_dims=2)
            l_feature = self.new_feature_layer(l_obs_inputs)
            l_rnn = self.new_rnn_layer(
                l_feature,
                num_units=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
            )
            l_out = L.TemporalUnflattenLayer(
                L.DenseLayer(
                    L.TemporalFlattenLayer(l_rnn),
                    num_units=action_dim + 1,
                    nonlinearity=None,
                ),
                ref_layer=l_obs_inputs[0]
            )

            if isinstance(env_spec.action_space, Box):
                l_log_std = L.ParamLayer(l_obs_inputs[0], num_units=action_dim)
                l_params = [l_log_std]
            else:
                l_params = []

            self.l_obs_inputs = l_obs_inputs
            self.l_feature = l_feature
            self.l_rnn = l_rnn
            self.l_out = l_out
            self.l_params = l_params

            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.state_dim = l_rnn.state_dim

            with tf.variable_scope("retfilter"):
                self.ret_rms = RunningMeanStd()

            self.prev_states = None
            if isinstance(env_spec.action_space, Discrete):
                self.dist = RecurrentCategorical(env_spec.action_space.n)
            elif isinstance(env_spec.action_space, Box):
                self.dist = RecurrentDiagonalGaussian(env_spec.action_space.flat_dim)
            else:
                raise NotImplementedError

            flat_obs_vars = self.new_obs_vars(extra_dims=1)
            obs_vars = [tf.expand_dims(var, 0) for var in flat_obs_vars]
            prev_state_var = tf.placeholder(
                dtype=tf.float32,
                shape=(None, l_rnn.cell.state_size),
                name="prev_state"
            )
            recurrent_state_output = dict()

            dist_info_vars = self.dist_info_sym(
                obs_vars,
                recurrent_state={l_rnn: prev_state_var},
                recurrent_state_output=recurrent_state_output,
            )

            self.f_step = tensor_utils.compile_function(
                inputs=flat_obs_vars + [prev_state_var],
                outputs=[dist_info_vars[k][0] for k in self.dist.dist_info_keys] + \
                        [
                            dist_info_vars["v"][0],
                            recurrent_state_output[l_rnn]
                        ],
            )

            LayersPowered.__init__(self, [l_out] + l_params)

    def new_obs_input_layers(self, extra_dims=2):
        return [
            L.InputLayer(
                shape=(None,) * extra_dims + (self.observation_space.flat_dim,),
                name="input"
            )
        ]

    def new_feature_layer(self, input_layers):
        if len(input_layers) == 1:
            return input_layers[0]
        return L.concat(input_layers, axis=-1)

    def new_rnn_layer(self, feature_layer, num_units, hidden_nonlinearity):
        return LL.TfRNNLayer(
            incoming=feature_layer,
            num_units=num_units,
            cell=GRUCell(num_units=num_units, activation=hidden_nonlinearity, weight_normalization=True),
        )

    def dist_info_sym(self, obs_vars, state_info_vars=None, **kwargs):
        out, *params = L.get_output(
            [self.l_out] + self.l_params,
            dict(zip(self.l_obs_inputs, obs_vars)),
            **kwargs
        )
        dist_var = out[:, :, :self.action_dim]
        v_var = out[:, :, self.action_dim:]
        v_var = v_var * self.ret_rms.std + self.ret_rms.mean
        dist_info = dict(v=v_var)
        if isinstance(self.action_space, Discrete):
            prob_var = tensor_utils.temporal_unflatten_sym(
                tf.nn.softmax(tensor_utils.temporal_flatten_sym(dist_var)),
                ref_var=obs_vars[0]
            )
            dist_info["prob"] = prob_var
        elif isinstance(self.action_space, Box):
            mean_var = dist_var
            log_std_var = params[0]
            dist_info["mean"] = mean_var
            dist_info["log_std"] = log_std_var
        else:
            raise NotImplementedError
        return dist_info

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_states is None or len(dones) != len(self.prev_states):
            self.prev_states = np.zeros((len(dones), self.state_dim))

        if np.any(dones):
            self.prev_states[dones] = 0.

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def new_obs_vars(self, extra_dims=2):
        return [
            tf.placeholder(
                dtype=tf.float32,
                shape=(None,) * extra_dims + (self.observation_space.flat_dim,),
                name="obs"
            )
        ]

    def preprocess(self, observations):
        return (self.observation_space.flatten_n(observations),)

    def get_actions(self, observations):
        obs_inputs = self.preprocess(observations)
        prev_state = self.prev_states
        *dist_infos, vs, state_vec = self.f_step(*obs_inputs, self.prev_states)

        agent_info = dict(v=vs, prev_state=prev_state, state=state_vec)
        if isinstance(self.action_space, Discrete):
            probs, = dist_infos
            actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
            agent_info["prob"] = probs
        elif isinstance(self.action_space, Box):
            means, log_stds = dist_infos
            actions = means + np.random.normal() * np.exp(log_stds)
            agent_info["mean"] = means
            agent_info["log_std"] = log_stds
        else:
            raise NotImplementedError

        self.prev_states = state_vec
        return actions, agent_info

    @property
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    def get_params_internal(self, **tags):
        params = LayersPowered.get_params_internal(self, **tags)
        if not tags.get('trainable', False):
            return params + self.ret_rms.variables
        else:
            return params

    def log_diagnostics(self, paths, logger=None):
        if logger is None:
            logger = root_logger
        if isinstance(self.action_space, Box):
            log_stds = tensor_utils.concat_tensor_list([p["agent_infos"]["log_std"] for p in paths])
            stds = np.exp(log_stds)
            avg_std = np.mean(stds, axis=0)
            logger.record_tabular("AveragePolicyStd", np.mean(avg_std), op=np.mean)
            for idx, std in enumerate(avg_std):
                logger.record_tabular("AveragePolicyStd[%d]" % idx, std, op=np.mean)


def rollout_seg_generator(env, agent, batch_size, max_path_length=np.inf):
    """
    Rollout a segment of a trajectory.

    Inspired from John's code.
    """
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    dones = []
    obs = env.reset()
    done = False
    agent.reset()
    path_length = 0
    while True:
        a, agent_info = agent.get_action(obs)
        if done or len(observations) >= batch_size:
            yield dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                agent_infos=agent_infos,
                env_infos=env_infos,
                dones=dones,
                next_agent_info=agent_info,
            )
            if done:
                path_length = 0
            observations = []
            actions = []
            rewards = []
            agent_infos = []
            env_infos = []
            dones = []

        next_o, r, done, env_info = env.step(a)
        observations.append(obs)
        rewards.append(r)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1

        if path_length >= max_path_length:
            done = True

        dones.append(done)

        if done:
            next_o = env.reset()
            agent.reset()

        obs = next_o


class ParallelPPOSGD(object):
    def __init__(
            self,
            env,
            ac,
            batch_size=256,
            epoch_length=10000,
            discount=0.99,
            gae_lambda=0.95,
            max_path_length=np.inf,
            clip_lr=0.1,
            entropy_bonus=0.01,
            learning_rate=1e-3,
            opt_epochs=1,
            n_parallel=1,
            profile=False
    ):
        self.env = env
        self.ac = ac
        self.n_parallel = n_parallel

        if mpi_fork(n_parallel, profile) == "parent":
            sys.exit(0)

        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.max_path_length = max_path_length
        self.clip_lr = clip_lr
        self.entropy_bonus = entropy_bonus
        self.learning_rate = learning_rate
        self.opt_epochs = opt_epochs
        self.loss_vals = []
        self.paths = []
        self.path_segs = dict()
        self.logger = MpiLogger()

    def init_opt(self):
        obs_vars = self.ac.new_obs_vars(extra_dims=2)
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=2,
        )
        adv_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="advs")
        returns_var = tf.placeholder(dtype=tf.float32, shape=(None, None), name="returns")
        dist = self.ac.distribution
        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=(None, None) + shape, name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]
        prev_state_var = tf.placeholder(tf.float32, shape=(None, self.ac.l_rnn.state_dim), name="prev_state")

        recurrent_state_output = dict()

        dist_info_vars = self.ac.dist_info_sym(
            obs_vars, dict(),
            recurrent_state={self.ac.l_rnn: prev_state_var},
            recurrent_state_output=recurrent_state_output,
        )

        v_pred_var = dist_info_vars["v"]

        final_state = recurrent_state_output[self.ac.l_rnn]

        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        ent = tf.reduce_mean(dist.entropy_sym(dist_info_vars))

        adv_mean, adv_variance = tf.nn.moments(adv_var, axes=(0, 1))
        norm_adv_var = (adv_var - adv_mean) / (tf.sqrt(adv_variance) + 1e-8)

        surr1 = lr * norm_adv_var
        surr2 = tf.clip_by_value(lr, 1. - self.clip_lr, 1. + self.clip_lr) * norm_adv_var

        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        ret_z = (returns_var - self.ac.ret_rms.mean) / (self.ac.ret_rms.std + 1e-8)
        norm_v_pred_var = (v_pred_var - self.ac.ret_rms.mean) / (self.ac.ret_rms.std + 1e-8)

        vf_loss = 0.5 * tf.reduce_mean(tf.square(ret_z - norm_v_pred_var))
        policy_ent_loss = -self.entropy_bonus * ent
        mean_kl = tf.reduce_mean(kl)

        total_loss = policy_loss + vf_loss + policy_ent_loss
        losses = [policy_loss, vf_loss, policy_ent_loss, mean_kl, ent, tf.exp(ent)]
        loss_names = ["PolicyLoss", "VfLoss", "PolicyEntLoss", "MeanKL", "Entropy", "Perplexity"]

        inputs_dict = OrderedDict(
            [("obs_%d" % idx, var) for idx, var in enumerate(obs_vars)] + \
            [
                ("action", action_var),
                ("adv", adv_var),
                ("returns", returns_var),
            ] + \
            [(k, var) for k, var in zip(dist.dist_info_keys, old_dist_info_vars_list)] + \
            [
                ("prev_state", prev_state_var)
            ]
        )

        var_list = self.ac.get_params(trainable=True)

        adam = MpiAdam(var_list, stepsize=self.learning_rate)

        grads = tf.gradients(total_loss, var_list)
        grads = [g if g is not None else tf.zeros_like(v) for v, g in zip(var_list, grads)]

        flat_grads = tensor_utils.flatten_tensor_variables(grads)

        f_loss_grads = tensor_utils.compile_function(
            inputs=list(inputs_dict.values()),
            outputs=[flat_grads] + losses + [final_state],
        )

        return adam, f_loss_grads, loss_names, inputs_dict

    @profile
    def train(self):
        # Construct the optimization problem

        adam, f_loss_grads, loss_names, inputs_dict = self.init_opt()

        global_t = np.asarray(0)
        last_epoch = 0

        rank = MPI.COMM_WORLD.Get_rank()

        seed = ext.get_seed()
        if seed is None:
            seed = np.random.randint(low=0, high=np.iinfo(np.uint32).max)
        seed = seed + 10000 * int(rank)
        ext.set_seed(seed)

        if rank > 0:
            root_logger.disable()

        with tensor_utils.single_threaded_session() as sess, root_logger.prefix("Worker %d | " % rank):
            sess.run(tf.initialize_all_variables())

            # sync at the beginning of training
            adam.sync()
            # Do the single process version first

            gen = rollout_seg_generator(
                env=self.env,
                agent=self.ac,
                max_path_length=self.max_path_length,
                batch_size=self.batch_size
            )

            while True:
                path_seg = next(gen)

                obs_inputs = self.ac.preprocess(path_seg["observations"])
                path_seg["observations"] = self.env.observation_space.flatten_n(path_seg["observations"])
                path_seg["actions"] = actions = self.env.action_space.flatten_n(path_seg["actions"])
                path_seg["rewards"] = rewards = np.asarray(path_seg["rewards"])
                path_seg["agent_infos"] = agent_infos = tensor_utils.stack_tensor_dict_list(path_seg["agent_infos"])
                path_seg["env_infos"] = tensor_utils.stack_tensor_dict_list(path_seg["env_infos"])
                path_seg["dones"] = dones = np.asarray(path_seg["dones"])
                path_seg["notdones"] = notdones = 1 - dones
                path_seg["v_preds"] = v_preds = agent_infos["v"].flatten()

                next_agent_info = path_seg.pop("next_agent_info")

                self.record_path_seg(path_seg)

                if dones[-1]:
                    v_preds = np.append(v_preds, 0)
                else:
                    v_preds = np.append(v_preds, np.squeeze(next_agent_info["v"]))

                prev_state = agent_infos["prev_state"][0]

                notdones = np.append(notdones, 1)

                dist_info_list = [agent_infos[k] for k in self.ac.distribution.dist_info_keys]

                T = np.asarray(len(rewards))

                advs = np.empty(T, dtype=np.float32)
                last_adv = 0

                for t in reversed(range(T)):
                    notdone = notdones[t + 1]
                    delta = rewards[t] + self.discount * v_preds[t + 1] * notdone - v_preds[t]
                    advs[t] = last_adv = delta + self.discount * self.gae_lambda * notdone * last_adv

                # fitting target for value function
                returns = advs + v_preds[:-1]

                self.ac.ret_rms.update(returns)

                self.do_training(
                    inputs_dict=inputs_dict,
                    obs_inputs=obs_inputs,
                    actions=actions,
                    advs=advs,
                    returns=returns,
                    dist_info_list=dist_info_list,
                    prev_state=prev_state,
                    f_loss_grads=f_loss_grads,
                    adam=adam,
                )

                reduce_T = np.zeros_like(T)

                MPI.COMM_WORLD.Allreduce(T, reduce_T, op=MPI.SUM)

                global_t += reduce_T

                root_logger.log(str(global_t))

                if global_t // self.epoch_length > last_epoch:
                    # log!
                    self.log_diagnostics(
                        epoch=last_epoch,
                        loss_names=loss_names,
                        global_t=global_t
                    )
                    last_epoch = global_t // self.epoch_length

    def record_path_seg(self, path_seg):
        dones = path_seg["dones"]
        if np.any(dones):
            assert np.sum(dones) == 1
            assert dones[-1]

        for k, v in path_seg.items():
            if k in self.path_segs:
                self.path_segs[k].append(v)
            else:
                self.path_segs[k] = [v]

        if dones[-1]:
            path = dict()
            for k, vs in self.path_segs.items():
                if isinstance(vs[0], dict):
                    path[k] = tensor_utils.concat_tensor_dict_list(vs)
                else:
                    path[k] = tensor_utils.concat_tensor_list(vs)
            self.path_segs = dict()
            self.paths.append(path)

    def log_diagnostics(self, epoch, loss_names, global_t):
        for path in self.paths:
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)

        if len(self.paths) > 0:
            all_returns = np.concatenate([p["returns"] for p in self.paths])
            all_v_preds = np.concatenate([p["v_preds"] for p in self.paths])
            ev = special.explained_variance_1d(all_v_preds, all_returns)
        else:
            ev = np.nan
            all_returns = []
            all_v_preds = []

        # agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in self.paths])

        # ent = np.mean(self.ac.distribution.entropy(agent_infos))
        average_discounted_return = np.mean([p["returns"][0] for p in self.paths])

        undiscounted_returns = [np.sum(p["rewards"]) for p in self.paths]

        self.logger.record_tabular('Epoch', epoch, op=np.min)
        self.logger.record_tabular('GlobalT', global_t, op=np.min)
        self.logger.record_tabular('AverageDiscountedReturn',
                                   average_discounted_return, op=np.mean)
        self.logger.record_tabular('ExplainedVariance', ev, op=np.mean)
        self.logger.record_tabular('NumTrajs', len(self.paths), op=np.sum)
        self.logger.record_tabular_misc_stat('TrajLen', [len(p["rewards"]) for p in self.paths], placement='front')
        self.logger.record_tabular_misc_stat('Return', undiscounted_returns, placement='front')
        self.logger.record_tabular_misc_stat('AllReturns', all_returns, placement='front')
        self.logger.record_tabular_misc_stat('AllVPreds', all_v_preds, placement='front')

        for name, v in zip(loss_names, np.mean(self.loss_vals, axis=0)):
            self.logger.record_tabular(name, v, op=np.mean)

        self.env.log_diagnostics(self.paths, logger=self.logger)
        self.ac.log_diagnostics(self.paths, logger=self.logger)

        self.logger.dump_tabular()

        self.paths = []
        self.loss_vals = []

    @profile
    def do_training(
            self,
            inputs_dict,
            obs_inputs,
            actions,
            advs,
            returns,
            dist_info_list,
            prev_state,
            f_loss_grads,
            adam,
    ):
        temporal_inputs = OrderedDict(
            [("obs_%d" % idx, var) for idx, var in enumerate(obs_inputs)] + \
            [
                ("action", actions),
                ("adv", advs),
                ("returns", returns),
            ] + \
            [(k, var) for k, var in zip(self.ac.distribution.dist_info_keys, dist_info_list)] + \
            [
                ("prev_state", prev_state)
            ]
        )
        assert tuple(temporal_inputs.keys()) == tuple(inputs_dict.keys())

        temporal_inputs = [np.expand_dims(x, 0) for x in temporal_inputs.values()]

        adam.sync()

        for _ in range(self.opt_epochs):
            flat_g, *loss_vals, final_state = f_loss_grads(*temporal_inputs, prev_state)
            adam.update(flat_g)
            self.loss_vals.append(loss_vals)


def mpi_fork(n, profile=False):
    """
    Re-launches the current script with workers
    Returns -1 for parent, rank in {0,1,...,n-1} for children

    Taken from John's code.
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        if profile:
            command = [ "mpirun", "-np", str(n), "kernprof", "-o",
                        "/root/code/rllab/python.lprof", "-l",
                        ] + sys.argv
        else:
            command = ["mpirun", "-np", str(n), sys.executable] + sys.argv
        print(command)
        subprocess.check_call(command, env=env)
        return "parent"
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        root_logger.log("Worker %d launched" % rank)
        return "child"


class RNNLayerWrapper(object):
    def __init__(self, l_rnns):
        self.l_rnns = l_rnns

    @property
    def state_dim(self):
        return sum([l.state_dim for l in self.l_rnns])

    def split_kwargs(self, kwargs):
        kwargs_list = [dict() for _ in self.l_rnns]
        if 'recurrent_state' in kwargs and self in kwargs:
            joint_state = kwargs['recurrent_state'][self]
            states = self.split_state_sym(joint_state)
            for kwargs_i, state_i in zip(kwargs_list, states):
                kwargs_i['recurrent_state'] = state_i
        if 'recurrent_state_output' in kwargs:
            for kwargs_i in kwargs_list:
                kwargs_i['recurrent_state_output'] = dict()
        return kwargs_list

    def split_state_sym(self, state):
        dim = state.get_shape().ndims
        # assuming need to split the last dimension...
        slices = [slice(None) for _ in range(dim - 1)]
        cum_dim = 0
        states = []
        for l_rnn in self.l_rnns:
            dim = l_rnn.state_dim
            states.append(state[slices + [slice(cum_dim, cum_dim + dim)]])
            cum_dim += dim
        return states

    def join_states_sym(self, states):
        dims = [s.get_shape().ndims for s in states]
        assert len(set(dims)) == 1
        return tf.concat(dims[0] - 1, states)

    def join_states(self, states):
        return np.concatenate(states, axis=-1)

    def join_kwargs(self, kwargs_list, joint_kwargs):
        if 'recurrent_state_output' in joint_kwargs:
            states = []
            for kwargs_i, l_rnn in zip(kwargs_list, self.l_rnns):
                states.append(kwargs_i['recurrent_state_output'][l_rnn])
            joint_kwargs['recurrent_state_output'][self] = self.join_states_sym(states)


class SeparateActorCritic(Parameterized, Serializable):
    def __init__(self, ac):
        Serializable.quick_init(self, locals())
        with suppress_params_loading():
            self.policy_ac = Serializable.clone(ac, name="ac_policy")
            self.vf_ac = Serializable.clone(ac, name="ac_vf")
            self.ret_rms = self.policy_ac.ret_rms = self.vf_ac.ret_rms
        Parameterized.__init__(self)

    def get_actions(self, observations):
        actions, policy_agent_infos = self.policy_ac.get_actions(observations)
        _, vf_agent_infos = self.vf_ac.get_actions(observations)

        agent_infos = dict(policy_agent_infos)
        agent_infos["v"] = vf_agent_infos["v"]
        agent_infos["prev_state"] = self.l_rnn.join_states([
            policy_agent_infos["prev_state"],
            vf_agent_infos["prev_state"]
        ])
        agent_infos["state"] = self.l_rnn.join_states([
            policy_agent_infos["state"],
            vf_agent_infos["state"]
        ])
        return actions, agent_infos

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def new_obs_vars(self, extra_dims=2):
        return self.policy_ac.new_obs_vars(extra_dims=extra_dims)

    @cached_property
    def distribution(self):
        return self.policy_ac.distribution

    @cached_property
    def l_rnn(self):
        return RNNLayerWrapper([self.policy_ac.l_rnn, self.vf_ac.l_rnn])

    def dist_info_sym(self, obs_vars, state_info_vars, **kwargs):
        pol_kwargs, vf_kwargs = self.l_rnn.split_kwargs(kwargs)
        policy_dist = self.policy_ac.dist_info_sym(obs_vars, state_info_vars, **pol_kwargs)
        vf_dist = self.vf_ac.dist_info_sym(obs_vars, state_info_vars, **vf_kwargs)
        self.l_rnn.join_kwargs([pol_kwargs, vf_kwargs], kwargs)
        return dict(policy_dist, v=vf_dist["v"])

    def get_params_internal(self, **tags):
        vars = self.policy_ac.get_params_internal(**tags) + self.vf_ac.get_params_internal(**tags)
        return sorted(set(vars), key=lambda x: x.name)

    def reset(self):
        self.policy_ac.reset()
        self.vf_ac.reset()

    def preprocess(self, observations):
        return self.policy_ac.preprocess(observations)

    def log_diagnostics(self, *args, **kwargs):
        self.policy_ac.log_diagnostics(*args, **kwargs)


def main():
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
    from rllab.envs.box2d.cartpole_env import CartpoleEnv
    from rllab.envs.mujoco.swimmer_env import SwimmerEnv  # ()#box2d.cartpole_env import CartpoleEnv
    from rllab.envs.normalized_env import normalize
    from sandbox.rocky.neural_learner.envs.mab_env import MABEnv

    env = TfEnv(MultiEnv(
        wrapped_env=MABEnv(n_arms=10),
        n_episodes=500,
        episode_horizon=1,
        discount=0.99
    ))
    # env = TfEnv(normalize(CartpoleEnv()))
    # env = TfEnv(normalize(SwimmerEnv()))

    # ac = SeparateActorCritic(
    #     RNNActorCritic(name="ac", env_spec=env.spec)
    # )
    ac = RNNActorCritic(name="ac", env_spec=env.spec, hidden_dim=256)

    algo = ParallelPPOSGD(
        env=env,
        ac=ac,
        batch_size=500,  # 500,
        epoch_length=50000,
        opt_epochs=5,
        entropy_bonus=0.,
        max_path_length=500,
        n_parallel=32,
        gae_lambda=0.3,
    )

    algo.train()


# if __name__ == "__main__":
#     main()
