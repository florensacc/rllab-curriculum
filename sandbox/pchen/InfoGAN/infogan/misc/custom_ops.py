from contextlib import contextmanager

import enum
import prettytensor as pt
import tensorflow as tf
from prettytensor.pretty_tensor_class import Phase, join_pretty_tensors, PrettyTensor
import numpy as np

class CustomPhase(enum.Enum):
    """Some nodes are different depending on the phase of the graph construction.

    The standard phases are train, test and infer.
    """
    train = 1999
    test = 2999
    init = 3999


class conv_batch_norm(pt.VarStoreMethod):
    """Code modification of http://stackoverflow.com/a/33950177"""

    def __call__(self, input_layer, epsilon=1e-5, momentum=0.1, name="batch_norm",
                 in_dim=None, phase=Phase.train):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9)

        shape = input_layer.shape
        shp = in_dim or shape[-1]
        with tf.variable_scope(name) as scope:
            self.gamma = self.variable("gamma", [shp], init=tf.random_normal_initializer(1., 0.02))
            self.beta = self.variable("beta", [shp], init=tf.constant_initializer(0.))

            self.mean, self.variance = tf.nn.moments(input_layer.tensor, [0, 1, 2])
            # sigh...tf's shape system is so..
            self.mean.set_shape((shp,))
            self.variance.set_shape((shp,))
            self.ema_apply_op = self.ema.apply([self.mean, self.variance])

            if phase == Phase.train:
                with tf.control_dependencies([self.ema_apply_op]):
                    normalized_x = tf.nn.batch_norm_with_global_normalization(
                        input_layer.tensor, self.mean, self.variance, self.beta, self.gamma, epsilon,
                        scale_after_normalization=True)
            else:
                normalized_x = tf.nn.batch_norm_with_global_normalization(
                    x, self.ema.average(self.mean), self.ema.average(self.variance), self.beta,
                    self.gamma, epsilon,
                    scale_after_normalization=True)
            return input_layer.with_tensor(normalized_x, parameters=self.vars)


pt.Register(assign_defaults=('phase'))(conv_batch_norm)


@pt.Register(assign_defaults=('phase'))
class fc_batch_norm(conv_batch_norm):
    def __call__(self, input_layer, *args, **kwargs):
        ori_shape = input_layer.shape
        if ori_shape[0] is None:
            ori_shape[0] = -1
        new_shape = [ori_shape[0], 1, 1, ori_shape[1]]
        x = tf.reshape(input_layer.tensor, new_shape)
        normalized_x = super(self.__class__, self).__call__(input_layer.with_tensor(x), *args, **kwargs)  # input_layer)
        return normalized_x.reshape(ori_shape)


def leaky_rectify(x, leakiness=0.01):
    assert leakiness <= 1
    ret = tf.maximum(x, leakiness * x)
    # import ipdb; ipdb.set_trace()
    return ret


@pt.Register(
    assign_defaults=('activation_fn', 'custom_phase', ))
class custom_conv2d(pt.VarStoreMethod):
    def __call__(self, input_layer, output_dim,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, in_dim=None, padding='SAME', activation_fn=None,
                 name="conv2d", residual=False, custom_phase=CustomPhase.train):
        print("ignoring data init : %s" % custom_phase)
        with tf.variable_scope(name):
            w = self.variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
                              init=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_layer.tensor, w, strides=[1, d_h, d_w, 1], padding=padding)

            biases = self.variable('biases', [output_dim], init=tf.constant_initializer(0.0))
            # import ipdb; ipdb.set_trace()
            out = tf.nn.bias_add(conv, biases)
            if residual:
                out += input_layer.tensor
            if activation_fn:
                books = input_layer.bookkeeper
                out = layers.apply_activation(
                    books,
                    out,
                    activation_fn,

                )
            return input_layer.with_tensor(out, parameters=self.vars)


@pt.Register(
    assign_defaults=('activation_fn', 'custom_phase', 'wnorm', 'pixel_bias',
                     'var_scope',
    )
)
class custom_deconv2d(pt.VarStoreMethod):
    def __call__(self, input_layer, output_shape,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                 name="deconv2d", activation_fn=None, custom_phase=CustomPhase.train, init_scale=0.1,
                 wnorm=False, pixel_bias=False, var_scope=None, prefix="",
                 ):
        # print "data init: ", custom_phase
        output_shape[0] = input_layer.shape[0]
        books = input_layer.bookkeeper
        if var_scope:
            self.vars = books.var_mapping[var_scope]
        ts_output_shape = tf.pack(output_shape)
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = self.variable(prefix + 'w', [k_h, k_w, output_shape[-1], input_layer.shape[-1]],
                              init=tf.random_normal_initializer(stddev=stddev))
            if custom_phase == CustomPhase.init:
                w = w.initialized_value()

            if wnorm:
                w_norm = tf.nn.l2_normalize(w, [0,1,3])
            else:
                w_norm = w

            deconv = tf.nn.conv2d_transpose(input_layer, w_norm,
                                                output_shape=ts_output_shape,
                                                strides=[1, d_h, d_w, 1])
            bias_shp = [1,]+output_shape[1:] if pixel_bias else [1, 1, 1, output_shape[-1]]
            if wnorm:
                m_init, v_init = tf.nn.moments(deconv, [0,1,2], keep_dims=True)
                p_s_init = init_scale / tf.sqrt(v_init + 1e-9)
                w_scale = self.variable(
                    'w_scale',
                    [1,1,1,output_shape[-1]],
                    init=lambda *_,**__: p_s_init
                )
                biases = self.variable(
                    'biases',
                    bias_shp, #[1,1,1,output_shape[-1]],
                    init=lambda *_,**__: tf.ones(bias_shp)*-m_init*p_s_init
                )
                if custom_phase == CustomPhase.init:
                    biases = biases.initialized_value()
                    w_scale = w_scale.initialized_value()
                y = deconv*w_scale + biases
            else:
                biases = self.variable(
                    'biases',
                    bias_shp,
                    init=tf.constant_initializer(0.)
                )
                if custom_phase == CustomPhase.init:
                    biases = biases.initialized_value()
                y = deconv + biases
            if activation_fn:
                y = layers.apply_activation(
                    books,
                    y,
                    activation_fn,

                    )
            return input_layer.with_tensor(y, parameters=self.vars)


@pt.Register
class custom_fully_connected(pt.VarStoreMethod):
    def __call__(self, input_layer, output_size, scope=None, in_dim=None, stddev=0.02, bias_start=0.0):
        shape = input_layer.shape
        input_ = input_layer.tensor
        try:
            if len(shape) == 4:
                input_ = tf.reshape(input_, tf.pack([tf.shape(input_)[0], np.prod(shape[1:])]))
                input_.set_shape([None, np.prod(shape[1:])])
                shape = input_.get_shape().as_list()

            with tf.variable_scope(scope or "Linear"):
                matrix = self.variable("Matrix", [in_dim or shape[1], output_size], dt=tf.float32,
                                       init=tf.random_normal_initializer(stddev=stddev))
                bias = self.variable("bias", [output_size], init=tf.constant_initializer(bias_start))
                return input_layer.with_tensor(tf.matmul(input_, matrix) + bias, parameters=self.vars)
        except Exception:
            import ipdb;
            ipdb.set_trace()


import collections
import itertools

import tensorflow as tf

from prettytensor import functions
from prettytensor import layers
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor.pretty_tensor_class import DIM_REST
from prettytensor.pretty_tensor_class import DIM_SAME
from prettytensor.pretty_tensor_class import Phase
from prettytensor.pretty_tensor_class import PROVIDED


def debug(name, x):
    return x
    # return tf.Print(x, [fn(x) if callable(fn) else fn for fn in [
    #     name, "min", tf.reduce_min, "mean", tf.reduce_mean, "max", tf.reduce_max, "norm",
    #     lambda x: tf.reduce_mean(tf.square(x))
    # ]])


@pt.Register(assign_defaults=('activation_fn', 'l2loss', 'stddev', 'local_reparam', 'prior_std'))
class bayes_fully_connected(pt.VarStoreMethod):
    def __call__(self,
                 input_layer,
                 size,
                 activation_fn=None,
                 l2loss=None,
                 init=None,
                 stddev=None,
                 bias=True,
                 bias_init=tf.zeros_initializer,
                 transpose_weights=False,
                 prior_std=0.05,
                 local_reparam=False,
                 name=PROVIDED):
        """Adds the parameters for a fully connected layer and returns a tensor.
        The current head must be a rank 2 Tensor.
        Args:
          input_layer: The Pretty Tensor object, supplied.
          size: The number of neurons
          activation_fn: A tuple of (activation_function, extra_parameters). Any
            function that takes a tensor as its first argument can be used. More
            common functions will have summaries added (e.g. relu).
          l2loss: Set to a value greater than 0 to use L2 regularization to decay
            the weights.
          init: An optional initialization. If not specified, uses Xavier
            initialization.
          stddev: A standard deviation to use in parameter initialization.
          bias: Set to False to not have a bias.
          bias_init: The initializer for the bias or a Tensor.
          transpose_weights: Flag indicating if weights should be transposed;
            this is useful for loading models with a different shape.
          name: The name for this operation is also used to create/find the
            parameter variables.
        Returns:
          A Pretty Tensor handle to the layer.
        Raises:
          ValueError: if the head_shape is not rank 2  or the number of input nodes
          (second dim) is not known.
        """

        # TODO(eiderman): bias_init shouldn't take a constant and stddev shouldn't
        # exist.
        if input_layer.get_shape().ndims != 2:
            raise ValueError(
                'fully_connected requires a rank 2 Tensor with known second '
                'dimension: %s'
                % input_layer.get_shape())
        in_size = input_layer.shape[1]
        if input_layer.shape[1] is None:
            raise ValueError('Number of input nodes must be known.')
        books = input_layer.bookkeeper
        if init is None:
            if stddev is None:
                init = layers.he_init(in_size, size, activation_fn)
            else:
                tf.logging.warning(
                    'Passing `stddev` to initialize weight variable is deprecated and '
                    'will be removed in the future. Pass '
                    'tf.truncated_normal_initializer(stddev=stddev) or '
                    'tf.zeros_initializer to `init` instead.')
                if stddev:
                    init = tf.truncated_normal_initializer(stddev=stddev)
                else:
                    init = tf.zeros_initializer
        elif stddev is not None:
            raise ValueError('Do not set both init and stddev.')
        dtype = input_layer.tensor.dtype
        weight_shape = [size, in_size] if transpose_weights else [in_size, size]

        prior_std_rho = np.log(np.exp(prior_std) - 1)

        def rho2std(rho):
            std = tf.log(1 + tf.exp(rho))
            return std

        params_mean = self.variable(
            'weights_mean',
            weight_shape,
            init,
            dt=dtype
        )
        params_std_rho = self.variable(
            'weights_std_rho',
            weight_shape,
            prior_std_rho + np.zeros(weight_shape),
            dt=dtype
        )
        tf.add_to_collection("dist_vars", [params_mean, rho2std(params_std_rho)])

        if local_reparam:
            y_mean = tf.matmul(input_layer, params_mean, transpose_b=transpose_weights)
            y_var = tf.matmul(tf.square(input_layer), tf.square(rho2std(params_std_rho)), transpose_b=transpose_weights)
            if bias:
                if isinstance(bias_init, tf.compat.real_types):
                    bias_init = tf.constant_initializer(bias_init)
                bias_mean = self.variable(
                    'bias_mean',
                    [size],
                    bias_init,
                    dt=dtype)
                bias_std_rho = self.variable(
                    'bias_std_rho',
                    [size],
                    prior_std_rho + np.zeros([size]),
                    dt=dtype)
                tf.add_to_collection("dist_vars", [bias_mean, rho2std(bias_std_rho)])

                y_mean += bias_mean
                y_var += tf.square(rho2std(bias_std_rho))
            y = tf.square(y_var) * tf.random_normal(tf.shape(y_mean)) + y_mean
        else:
            params = tf.random_normal(weight_shape) * rho2std(params_std_rho) + params_mean
            y = tf.matmul(input_layer, params, transpose_b=transpose_weights)
            layers.add_l2loss(books, params, l2loss)
            if bias:
                if isinstance(bias_init, tf.compat.real_types):
                    bias_init = tf.constant_initializer(bias_init)
                bias_mean = self.variable(
                    'bias_mean',
                    [size],
                    bias_init,
                    dt=dtype)
                bias_std_rho = self.variable(
                    'bias_std_rho',
                    [size],
                    prior_std_rho + np.zeros([size]),
                    dt=dtype)
                bias = tf.random_normal([size]) * rho2std(bias_std_rho) + bias_mean
                y += bias

        if activation_fn is not None:
            if not isinstance(activation_fn, collections.Sequence):
                activation_fn = (activation_fn,)
            y = layers.apply_activation(
                books,
                y,
                activation_fn[0],
                activation_args=activation_fn[1:])
        books.add_histogram_summary(y, '%s/activations' % y.op.name)
        return input_layer.with_tensor(y, parameters=self.vars)


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = n_out / n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in / n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask

def get_linear_ar_mask_by_groups(n_in, n_out, ngroups, zerodiagonal=True):
    assert n_in % ngroups == 0 and n_out % ngroups == 0

    mask = np.ones([n_in, n_out], dtype=np.float32)
    j = n_in / ngroups
    k = n_out / ngroups

    for i in xrange(ngroups):
        mask[(i+1)*j:, i*k:(i+1)*k] = 0
        if zerodiagonal:
            mask[i*j:(i+1)*j, i*k:(i+1)*k] = 0
    return mask


@prettytensor.Register(
    assign_defaults=(
            'activation_fn', 'l2loss', 'stddev', 'ngroups',
            'wnorm', 'custom_phase', 'init_scale',
    ))
class arfc(prettytensor.VarStoreMethod):
    def __call__(self,
                 input_layer,
                 size,
                 activation_fn=None,
                 l2loss=None,
                 init=None,
                 stddev=None,
                 bias=True,
                 bias_init=tf.zeros_initializer,
                 transpose_weights=False,
                 ngroups=None,
                 zerodiagonal=False,
                 wnorm=False,
                 custom_phase=CustomPhase.train,
                 init_scale=0.1,
                 name=PROVIDED):
        """Adds the parameters for a fully connected layer and returns a tensor.
        The current head must be a rank 2 Tensor.
        Args:
          input_layer: The Pretty Tensor object, supplied.
          size: The number of neurons
          activation_fn: A tuple of (activation_function, extra_parameters). Any
            function that takes a tensor as its first argument can be used. More
            common functions will have summaries added (e.g. relu).
          l2loss: Set to a value greater than 0 to use L2 regularization to decay
            the weights.
          init: An optional initialization. If not specified, uses Xavier
            initialization.
          stddev: A standard deviation to use in parameter initialization.
          bias: Set to False to not have a bias.
          bias_init: The initializer for the bias or a Tensor.
          transpose_weights: Flag indicating if weights should be transposed;
            this is useful for loading models with a different shape.
          name: The name for this operation is also used to create/find the
            parameter variables.
        Returns:
          A Pretty Tensor handle to the layer.
        Raises:
          ValueError: if the head_shape is not rank 2  or the number of input nodes
          (second dim) is not known.
        """
        # print "arfc data init", custom_phase
        # TODO(eiderman): bias_init shouldn't take a constant and stddev shouldn't
        # exist.
        if input_layer.get_shape().ndims != 2:
            raise ValueError(
                'fully_connected requires a rank 2 Tensor with known second '
                'dimension: %s'
                % input_layer.get_shape())
        in_size = input_layer.shape[1]
        if input_layer.shape[1] is None:
            raise ValueError('Number of input nodes must be known.')
        books = input_layer.bookkeeper
        if init is None:
            if stddev is None:
                init = layers.he_init(in_size, size, activation_fn)
            else:
                tf.logging.warning(
                    'Passing `stddev` to initialize weight variable is deprecated and '
                    'will be removed in the future. Pass '
                    'tf.truncated_normal_initializer(stddev=stddev) or '
                    'tf.zeros_initializer to `init` instead.')
                if stddev:
                    init = tf.truncated_normal_initializer(stddev=stddev)
                else:
                    init = tf.zeros_initializer
        elif stddev is not None:
            raise ValueError('Do not set both init and stddev.')
        dtype = input_layer.tensor.dtype
        weight_shape = [size, in_size] if transpose_weights else [in_size, size]

        params = self.variable(
            'weights',
            weight_shape,
            init,
            dt=dtype)
        if ngroups:
            mask = get_linear_ar_mask_by_groups(in_size, size, ngroups, zerodiagonal=zerodiagonal)
        else:
            mask = get_linear_ar_mask(in_size, size, zerodiagonal=zerodiagonal)
        if transpose_weights:
            mask = tf.transpose(mask)
        # abusing bad design choice here. mask is actually being stored as a value (nontrainble)
        mask = self.variable(
            'masks',
            weight_shape,
            mask
        )
        assert not isinstance(mask, tf.Variable)
        if wnorm:
            params_init = params.initialized_value()
            normalized_init = tf.nn.l2_normalize(params_init, 0)
            y_init = tf.matmul(input_layer, normalized_init * mask, transpose_b=False)#transpose_weights)
            y_init = debug("y_init", y_init)
            y_mu, y_var = tf.nn.moments(y_init, [0], keep_dims=True)
            y_var = debug("y_var", y_var)
            p_s_init = init_scale / tf.sqrt(y_var + 1e-9)
            params_scale = self.variable(
                'weights_scale',
                [1, size],
                lambda *_,**__: p_s_init,
                dt=dtype
            )
            bias = self.variable(
                'bias',
                [1, size],
                lambda *_,**__: -y_mu * p_s_init,
                dt=dtype
            )
        else:
            bias = self.variable(
                'bias',
                [1, size],
                tf.constant_initializer(0.),
                dt=dtype
            )

        # real stuff
        if custom_phase == CustomPhase.init:
            params = params.initialized_value()
            bias = bias.initialized_value()
            if wnorm:
                params_scale = params_scale.initialized_value()

        if wnorm:
            normalized = tf.nn.l2_normalize(params, 0)
        else:
            normalized = params
        inp = input_layer.tensor
        y = tf.matmul(inp, normalized*mask, transpose_b=False)#transpose_weights)
        if wnorm:
            y = y*params_scale
        y = y+bias

        layers.add_l2loss(books, params, l2loss)
        if activation_fn is not None:
            if not isinstance(activation_fn, collections.Sequence):
                activation_fn = (activation_fn,)
            y = layers.apply_activation(
                books,
                y,
                activation_fn[0],
                activation_args=activation_fn[1:])
        # books.add_histogram_summary(y, '%s/activations' % y.op.name)
        return input_layer.with_tensor(y, parameters=self.vars)

from prettytensor.pretty_tensor_image_methods import *
@prettytensor.Register(
    assign_defaults=(
            'activation_fn', 'l2loss', 'stddev', 'batch_normalize',
            'custom_phase', 'wnorm', 'pixel_bias', 'var_scope',
    )
)
class conv2d_mod(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               kernel,
               depth,
               activation_fn=None,
               stride=None,
               l2loss=None,
               init=None,
               stddev=None,
               bias=True,
               bias_init=tf.zeros_initializer,
               edges=PAD_SAME,
               batch_normalize=False,
               residual=False,
               custom_phase=CustomPhase.train,
               wnorm=False,
               pixel_bias=False,
               scale_init=0.1,
               var_scope=None,
               prefix="",
               name=PROVIDED
               ):
    """Adds a convolution to the stack of operations.
    The current head must be a rank 4 Tensor.
    Args:
      input_layer: The chainable object, supplied.
      kernel: The size of the patch for the pool, either an int or a length 1 or
        2 sequence (if length 1 or int, it is expanded).
      depth: The depth of the new Tensor.
      activation_fn: A tuple of (activation_function, extra_parameters). Any
        function that takes a tensor as its first argument can be used. More
        common functions will have summaries added (e.g. relu).
      stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
        int, length 1 or 2, the stride in the first and last dimensions are 1.
      l2loss: Set to a value greater than 0 to use L2 regularization to decay
        the weights.
      init: An optional initialization. If not specified, uses Xavier
        initialization.
      stddev: A standard deviation to use in parameter initialization.
      bias: Set to False to not have a bias.
      bias_init: An initializer for the bias or a Tensor.
      edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
        the output size and only uses valid input pixels.
      batch_normalize: Supply a BatchNormalizationArguments to set the
        parameters for batch normalization.
      name: The name for this operation is also used to create/find the
        parameter variables.
    Returns:
      Handle to the generated layer.
    Raises:
      ValueError: If head is not a rank 4 tensor or the  depth of the input
        (4th dim) is not known.
    """
    # print "data init: ", custom_phase
    if input_layer.get_shape().ndims != 4:
      raise ValueError('conv2d requires a rank 4 Tensor with a known depth %s' %
                       input_layer.get_shape())
    if input_layer.shape[3] is None:
      raise ValueError('Input depth must be known')
    from prettytensor.pretty_tensor_image_methods import _kernel
    kernel = _kernel(kernel)
    from prettytensor.pretty_tensor_image_methods import _stride
    stride = _stride(stride)
    size = [kernel[0], kernel[1], input_layer.shape[3], depth]

    books = input_layer.bookkeeper

    if var_scope:
        old_vars = self.vars
        new_vars = books.var_mapping[var_scope]
        self.vars = new_vars
        # import ipdb; ipdb.set_trace()
    assert init is None
    init = tf.random_normal_initializer(stddev=0.02)
    dtype = input_layer.tensor.dtype
    params = self.variable(prefix + 'weights', size, init, dt=dtype)
    if custom_phase == CustomPhase.init:
        params = params.initialized_value()
    params_norm = tf.nn.l2_normalize(params, [0,1,2]) if wnorm else params
    y = tf.nn.conv2d(input_layer, params_norm, stride, edges)
    layers.add_l2loss(books, params, l2loss)

    out_w = int(y.get_shape()[1])
    out_h = int(y.get_shape()[2])
    bias_shp = [1, out_w, out_h, depth] if pixel_bias else [1, 1, 1, depth]
    if wnorm:
        m_init, v_init = tf.nn.moments(y, [0, 1, 2], keep_dims=True)
        p_s_init = scale_init / tf.sqrt(v_init + 1e-9)
        if var_scope:
            self.vars = old_vars
        params_scale = self.variable(
            'weights_scale',
            [1, 1, 1, depth],
            lambda *_,**__: p_s_init,
            dt=dtype
        )
        b = self.variable(
            'bias',
            bias_shp,
            lambda *_,**__: tf.ones(bias_shp)*-m_init*p_s_init,
            dt=dtype
        )
        if var_scope:
            self.vars = new_vars
        if custom_phase == CustomPhase.init:
            b = b.initialized_value()
            params_scale = params_scale.initialized_value()
        y *= params_scale
        y += b
    else:
        b = self.variable(
            'bias',
            bias_shp,
            tf.constant_initializer(0.),
            dt=dtype
        )
        if custom_phase == CustomPhase.init:
            b = b.initialized_value()
        y += b

    books.add_scalar_summary(
        tf.reduce_mean(layers.spatial_slice_zeros(y)),
        '%s/zeros_spatial' % y.op.name)
    y = pretty_tensor_normalization_methods.batch_normalize_with_arguments(
        y, batch_normalize)
    if residual:
        y += input_layer
    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(books,
                                  y,
                                  activation_fn[0],
                                  activation_args=activation_fn[1:])
    books.add_histogram_summary(y, '%s/activations' % y.op.name)
    return input_layer.with_tensor(y, parameters=self.vars)

# @prettytensor.Register(
#     assign_defaults=('activation_fn', 'l2loss', 'stddev', 'batch_normalize'))
# class deconv2d_mod(prettytensor.VarStoreMethod):
#
#     def __call__(self,
#                  input_layer,
#                  kernel,
#                  depth,
#                  img_shp,
#                  activation_fn=None,
#                  stride=None,
#                  l2loss=None,
#                  init=None,
#                  stddev=None,
#                  bias=True,
#                  bias_init=tf.zeros_initializer,
#                  edges=PAD_SAME,
#                  batch_normalize=False,
#                  name=PROVIDED):
#         """Adds a convolution to the stack of operations.
#         The current head must be a rank 4 Tensor.
#         Args:
#           input_layer: The chainable object, supplied.
#           kernel: The size of the patch for the pool, either an int or a length 1 or
#             2 sequence (if length 1 or int, it is expanded).
#           depth: The depth of the new Tensor.
#           activation_fn: A tuple of (activation_function, extra_parameters). Any
#             function that takes a tensor as its first argument can be used. More
#             common functions will have summaries added (e.g. relu).
#           stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
#             int, length 1 or 2, the stride in the first and last dimensions are 1.
#           l2loss: Set to a value greater than 0 to use L2 regularization to decay
#             the weights.
#           init: An optional initialization. If not specified, uses Xavier
#             initialization.
#           stddev: A standard deviation to use in parameter initialization.
#           bias: Set to False to not have a bias.
#           bias_init: An initializer for the bias or a Tensor.
#           edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
#             the output size and only uses valid input pixels.
#           batch_normalize: Supply a BatchNormalizationArguments to set the
#             parameters for batch normalization.
#           name: The name for this operation is also used to create/find the
#             parameter variables.
#         Returns:
#           Handle to the generated layer.
#         Raises:
#           ValueError: If head is not a rank 4 tensor or the  depth of the input
#             (4th dim) is not known.
#         """
#         if input_layer.get_shape().ndims != 4:
#             raise ValueError('conv2d requires a rank 4 Tensor with a known depth %s' %
#                              input_layer.get_shape())
#         if input_layer.shape[3] is None:
#             raise ValueError('Input depth must be known')
#         from prettytensor.pretty_tensor_image_methods import _kernel
#         kernel = _kernel(kernel)
#         from prettytensor.pretty_tensor_image_methods import _stride
#         stride = _stride(stride)
#         size = [kernel[0], kernel[1], depth, input_layer.shape[3],]
#
#         books = input_layer.bookkeeper
#         if init is None:
#             if stddev is None:
#                 patch_size = size[0] * size[1]
#                 init = layers.he_init(size[2] * patch_size, size[3] * patch_size,
#                                       activation_fn)
#             else:
#                 tf.logging.warning(
#                     'Passing `stddev` to initialize weight variable is deprecated and '
#                     'will be removed in the future. Pass '
#                     'tf.truncated_normal_initializer(stddev=stddev) or '
#                     'tf.zeros_initializer to `init` instead.')
#                 if stddev:
#                     init = tf.truncated_normal_initializer(stddev=stddev)
#                 else:
#                     init = tf.zeros_initializer
#         elif stddev is not None:
#             raise ValueError('Do not set both init and stddev.')
#         dtype = input_layer.tensor.dtype
#         params = self.variable('weights', size, init, dt=dtype).initialized_value()
#
#         out_shp = [input_layer.shape[0]] + img_shp + [depth]
#         ts_output_shape = tf.pack(out_shp)
#
#         y = tf.nn.conv2d_transpose(input_layer, params, ts_output_shape, stride, edges)
#         layers.add_l2loss(books, params, l2loss)
#         if bias:
#             y += self.variable('bias', [size[-1]], bias_init, dt=dtype).initialized_value()
#         books.add_scalar_summary(
#             tf.reduce_mean(layers.spatial_slice_zeros(y)),
#             '%s/zeros_spatial' % y.op.name)
#         y = pretty_tensor_normalization_methods.batch_normalize_with_arguments(
#             y, batch_normalize)
#         if activation_fn is not None:
#             if not isinstance(activation_fn, collections.Sequence):
#                 activation_fn = (activation_fn,)
#             y = layers.apply_activation(books,
#                                         y,
#                                         activation_fn[0],
#                                         activation_args=activation_fn[1:])
#         books.add_histogram_summary(y, '%s/activations' % y.op.name)
#         return input_layer.with_tensor(y, parameters=self.vars)

@prettytensor.Register(assign_defaults=('activation_fn', 'l2loss', 'stddev', 'ngroups', 'custom_phase',
    "wnorm"))
class wnorm_fc(prettytensor.VarStoreMethod):
    def __call__(self,
                 input_layer,
                 size,
                 init=None,
                 activation_fn=None,
                 l2loss=None,
                 # stddev=None,
                 # bias=True,
                 # bias_init=tf.zeros_initializer,
                 # transpose_weights=False,
                 init_scale=1.0,
                 custom_phase=CustomPhase.train,
                 wnorm=False,
                 name=PROVIDED,
                 ):
        """Adds the parameters for a fully connected layer and returns a tensor.
        The current head must be a rank 2 Tensor.
        Args:
          input_layer: The Pretty Tensor object, supplied.
          size: The number of neurons
          activation_fn: A tuple of (activation_function, extra_parameters). Any
            function that takes a tensor as its first argument can be used. More
            common functions will have summaries added (e.g. relu).
          l2loss: Set to a value greater than 0 to use L2 regularization to decay
            the weights.
          init: An optional initialization. If not specified, uses Xavier
            initialization.
          stddev: A standard deviation to use in parameter initialization.
          bias: Set to False to not have a bias.
          bias_init: The initializer for the bias or a Tensor.
          transpose_weights: Flag indicating if weights should be transposed;
            this is useful for loading models with a different shape.
          name: The name for this operation is also used to create/find the
            parameter variables.
        Returns:
          A Pretty Tensor handle to the layer.
        Raises:
          ValueError: if the head_shape is not rank 2  or the number of input nodes
          (second dim) is not known.
        """
        # TODO(eiderman): bias_init shouldn't take a constant and stddev shouldn't
        # exist.
        # print "data init: ", custom_phase
        if input_layer.get_shape().ndims != 2:
            raise ValueError(
                'fully_connected requires a rank 2 Tensor with known second '
                'dimension: %s'
                % input_layer.get_shape())
        in_size = input_layer.shape[1]
        if input_layer.shape[1] is None:
            raise ValueError('Number of input nodes must be known.')
        books = input_layer.bookkeeper
        assert init is None
        init = tf.random_normal_initializer(0., 0.05)
        dtype = input_layer.tensor.dtype
        # weight_shape = [size, in_size] if transpose_weights else [in_size, size]
        weight_shape = [in_size, size]

        params = self.variable(
            'weights',
            weight_shape,
            init,
            dt=dtype
        )
        if wnorm:
            params_init = params.initialized_value()
            normalized_init = tf.nn.l2_normalize(params_init, 0)
            y_init = tf.matmul(input_layer, normalized_init, transpose_b=False)#transpose_weights)
            y_init = debug("y_init", y_init)
            y_mu, y_var = tf.nn.moments(y_init, [0], keep_dims=True)
            y_var = debug("y_var", y_var)
            p_s_init = init_scale / tf.sqrt(y_var + 1e-9)
            params_scale = self.variable(
                'weights_scale',
                [1, size],
                lambda *_,**__: p_s_init,
                dt=dtype
            )
            bias = self.variable(
                'bias',
                [1, size],
                lambda *_,**__: -y_mu * p_s_init,
                dt=dtype
            )
        else:
            bias = self.variable(
                'bias',
                [1, size],
                tf.constant_initializer(0.),
                dt=dtype
            )

        # real stuff
        if custom_phase == CustomPhase.init:
            params = params.initialized_value()
            bias = bias.initialized_value()
            if wnorm:
                params_scale = params_scale.initialized_value()

        if wnorm:
            normalized = tf.nn.l2_normalize(params, 0)
        else:
            normalized = params
        inp = input_layer.tensor
        y = tf.matmul(inp, normalized, transpose_b=False)#transpose_weights)
        if wnorm:
            y = y*params_scale
        y = y+bias

        layers.add_l2loss(books, params, l2loss)
        if activation_fn is not None:
            if not isinstance(activation_fn, collections.Sequence):
                activation_fn = (activation_fn,)
            y = layers.apply_activation(
                books,
                y,
                activation_fn[0],
                activation_args=activation_fn[1:])
        # books.add_histogram_summary(y, '%s/activations' % y.op.name)
        return input_layer.with_tensor(y, parameters=self.vars)

@prettytensor.Register(assign_defaults=('activation_fn', 'custom_phase'))
class nl(prettytensor.VarStoreMethod):
    def __call__(self,
                 input_layer,
                 activation_fn=None,
                 custom_phase=None,
                 ):
        # print("nl using phase %s " % custom_phase)
        books = input_layer.bookkeeper
        y = input_layer.tensor
        if activation_fn is not None:
            if not isinstance(activation_fn, collections.Sequence):
                activation_fn = (activation_fn,)
            y = layers.apply_activation(
                books,
                y,
                activation_fn[0],
                activation_args=activation_fn[1:])
        # books.add_histogram_summary(y, '%s/activations' % y.op.name)
        return input_layer.with_tensor(y, parameters=self.vars)

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.

    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

def resize_nearest_neighbor(x, scale):
    input_shape = map(int, x.get_shape().as_list())
    size = [int(input_shape[1] * scale), int(input_shape[2] * scale)]
    x = tf.image.resize_nearest_neighbor(x, size)
    return x

def resconv_v1(l_in, kernel, nch, stride=1, add_coeff=0.1, keep_prob=1., nn=False):
    seq = l_in.sequential()
    with seq.subdivide_with(2, tf.add_n) as [blk, origin]:
        blk.conv2d_mod(kernel, nch, stride=stride, prefix="pre")
        blk.custom_dropout(keep_prob)
        blk.conv2d_mod(kernel, nch, activation_fn=None, prefix="post")
        blk.apply(lambda x: x*add_coeff)
        if nn:
            origin.apply(resize_nearest_neighbor, 1./stride)
            origin.apply(lambda o: tf.tile(o, [1,1,1,nch/int(o.get_shape()[3])]))
        else:
            if stride != 1:
                origin.conv2d_mod(kernel, nch, stride=stride, activation_fn=None)
    return seq.as_layer().nl()

def resdeconv_v1(l_in, kernel, nch, out_wh, add_coeff=0.1, keep_prob=1., nn=False):
    seq = l_in.sequential()
    with seq.subdivide_with(2, tf.add_n) as [blk, origin]:
        blk.custom_deconv2d([0]+out_wh+[nch], k_h=kernel, k_w=kernel, prefix="de_pre")
        blk.custom_dropout(keep_prob)
        blk.conv2d_mod(kernel, nch, activation_fn=None, prefix="post")
        blk.apply(lambda x: x*add_coeff)
        if nn:
            origin.apply(tf.image.resize_nearest_neighbor, out_wh)
            origin.apply(lambda o: tf.reshape(o, [tf.shape(o)[0]]+out_wh+[nch, -1]))
            # origin.reshape([-1,]+out_wh+[nch, 2])
            origin.apply(tf.reduce_mean, [4],)
        else:
            origin.custom_deconv2d([0]+out_wh+[nch], k_h=kernel, k_w=kernel, activation_fn=None, prefix="de_pre")
    return seq.as_layer().nl()

def logsumexp(x):
    x_max = tf.reduce_max(x, [1], keep_dims=True)
    return tf.reshape(x_max, [-1]) + tf.log(tf.reduce_sum(tf.exp(x - x_max), [1]))

from prettytensor.bookkeeper import Bookkeeper
from collections import defaultdict
class CustomBookkeeper(Bookkeeper):
    def __init__(self, **kw):
        super(CustomBookkeeper, self).__init__(**kw)
        self.var_mapping = defaultdict(dict)

prettytensor.bookkeeper.BOOKKEEPER_FACTORY = CustomBookkeeper

@prettytensor.Register(assign_defaults=['custom_phase','model_avg'],)
def custom_dropout(
        input_layer,
        keep_prob,
        custom_phase=CustomPhase.train,
        model_avg=False,
        name=PROVIDED
):
    """Aplies dropout if this is in the train phase."""
    if keep_prob == 1.:
        return input_layer
    # print("dropout called with phase: %s" % custom_phase)
    if custom_phase == CustomPhase.test and model_avg:
        print("Using model averaging %s" % keep_prob)
        return input_layer * keep_prob
    else:
        return tf.nn.dropout(input_layer, keep_prob, name=name)

