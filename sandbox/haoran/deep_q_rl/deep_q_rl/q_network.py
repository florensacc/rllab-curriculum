"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""
import lasagne
import numpy as np
import theano
import theano.tensor as T
from .updates import deepmind_rmsprop
from rllab.core.serializable import Serializable
from rllab.misc import logger

class DeepQLearner(Serializable):
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 use_double, batch_size, network_type, conv_type, update_rule,
                 batch_accumulator, input_scale=255.0, network_args=dict(),
                 eta=0):
        Serializable.quick_init(self,locals())

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.use_double = use_double
        self.eta = eta

        # Using Double DQN is pointless without periodic freezing
        if self.use_double:
            assert self.freeze_interval > 0


        self.update_counter = 0

        self.l_out = self.build_network(
            network_type, conv_type,
            input_width, input_height,
            num_actions, num_frames, batch_size,
            network_args,
        )
        if self.freeze_interval > 0:
            self.next_l_out = self.build_network(
                network_type, conv_type,
                input_width, input_height,
                num_actions, num_frames, batch_size,
                network_args,
            )
            self.reset_q_hat()

        states = T.tensor4('states')
        next_states = T.tensor4('next_states') # "next" = "target"
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')
        returns = T.col('returns')

        self.states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.returns_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)

        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    next_states / input_scale)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    next_states / input_scale)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        if self.use_double:
            maxaction = T.argmax(q_vals, axis=1, keepdims=False)
            temptargets = next_q_vals[T.arange(batch_size),maxaction].reshape((-1, 1))
            bootstrap_target = (rewards +
                          (T.ones_like(terminals) - terminals) *
                          self.discount * temptargets)
        else:
            bootstrap_target = (rewards +
                          (T.ones_like(terminals) - terminals) *
                          self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        target = (1-self.eta) * bootstrap_target + self.eta * returns
        diff = target - q_vals[T.arange(batch_size),
                               actions.reshape((-1,))].reshape((-1, 1))

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            #
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(
            self.l_out,
            trainable=True,
        )
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared,
            returns: self.returns_shared,
        }
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens,allow_input_downcast=True)
        self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared},allow_input_downcast=True)

    def build_network(self, network_type, conv_type, input_width, input_height,
                      output_dim, num_frames, batch_size, extra_args=dict()):
        if conv_type == "cuda":
            from lasagne.layers.cuda_convnet import Conv2DCCLayer as conv_layer
            logger.log("Using lasagne.layers.cuda_convnet.Conv2DCCLayer to build conv layers.")
        elif conv_type == "cudnn":
            from lasagne.layers.dnn import Conv2DDNNLayer as conv_layer
            logger.log("Using lasagne.layers.dnnConv2DNNLayer to build conv layers.")
        else:
            from lasagne.layers import Conv2DLayer as conv_layer
            logger.log("Using lasagne.layers.Conv2DLayer to build conv layers.")

        if network_type == "nature":
            network_builder = self.build_nature_network

        elif network_type == "nips":
            network_builder = self.build_nips_network

        elif network_type == "linear":
            network_builder = self.build_linear_network

        elif network_type == "mlp":
            network_builder = self.build_mlp_network

        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

        return network_builder(
                input_width, input_height,
                output_dim, num_frames,
                batch_size, conv_layer,**extra_args)


    def train(self, states, actions, rewards, next_states, terminals, returns):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)
        returns - b x 1 numpy array of total total rewards

        Returns: average loss
        """
        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        self.returns_shared.set_value(returns)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss, _ = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state):
        states = np.zeros((self.batch_size, self.num_frames, self.input_height,
                           self.input_width), dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        return self._q_vals()[0]

    def choose_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

    def build_nature_network(
            self,
            input_width,
            input_height,
            output_dim,
            num_frames,
            batch_size,
            conv_layer,
        ):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = conv_layer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1),
        )

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
        )

        l_conv3 = conv_layer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out





    def build_nips_network(self, input_width, input_height, output_dim,
                           num_frames, batch_size, conv_layer):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = conv_layer(
            l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
        )

        l_conv2 = conv_layer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_dim,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out


    def build_linear_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size, conv_layer):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        For one-hot discrete states, this is exactly tabular look-up, with zero initial Q-values.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=None
        )

        return l_out

    def build_mlp_network(self, input_width, input_height, output_dim,
                             num_frames, batch_size, conv_layer, hidden_sizes, batch_norm):
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        from rllab.core.network import MLP
        network = MLP(
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=lasagne.nonlinearities.tanh,
            output_nonlinearity=None,
            input_layer=l_in,
            batch_norm=batch_norm,
        )
        return network.output_layer

    def get_param_values(self):
        params = lasagne.layers.get_all_param_values(self.l_out)
        return params


# def main():
#     net = DeepQLearner(84, 84, 16, 4, .99, .00025, .95, .95, 10000, False,
#                        32, 'nature_cuda')
#     # this set of parameters is never used. Instead, see run_nips, run_nature, or run_double
#
#
# if __name__ == '__main__':
#     main()
