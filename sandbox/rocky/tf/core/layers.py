from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import math
import tensorflow as tf
from collections import OrderedDict
from collections import deque
from itertools import chain
from inspect import getargspec
from difflib import get_close_matches
from warnings import warn


def create_param(spec, shape, name, trainable=True, regularizable=True):
    assert hasattr(spec, '__call__')
    if regularizable:
        # use the default regularizer
        regularizer = None
    else:
        # do not regularize this variable
        regularizer = lambda _: tf.constant(0.)
    return tf.get_variable(
        name=name, shape=shape, initializer=spec, trainable=trainable,
        regularizer=regularizer
    )


class Layer(object):
    def __init__(self, incoming, name, **kwargs):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming
        self.params = OrderedDict()
        self.name = name
        self.get_output_kwargs = []

        if any(d is not None and d <= 0 for d in self.input_shape):
            raise ValueError((
                                 "Cannot create Layer with a non-positive input_shape "
                                 "dimension. input_shape=%r, self.name=%r") % (
                                 self.input_shape, self.name))

    @property
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shape)
        if any(isinstance(s, (tf.Variable, tf.Tensor)) for s in shape):
            raise ValueError("%s returned a symbolic output shape from its "
                             "get_output_shape_for() method: %r. This is not "
                             "allowed; shapes must be tuples of integers for "
                             "fixed-size dimensions and Nones for variable "
                             "dimensions." % (self.__class__.__name__, shape))
        return shape

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError

    def get_output_for(self, input, **kwargs):
        raise NotImplementedError

    def add_param(self, spec, shape, name, **tags):
        with tf.variable_scope(self.name):
            tags['trainable'] = tags.get('trainable', True)
            tags['regularizable'] = tags.get('regularizable', True)
            param = create_param(spec, shape, name, **tags)
            self.params[param] = set(tag for tag, value in tags.items() if value)
            return param

    def get_params(self, **tags):
        result = list(self.params.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        return result


class InputLayer(Layer):
    def __init__(self, shape, name, input_var=None, **kwargs):
        super(InputLayer, self).__init__(shape, name, **kwargs)
        self.shape = shape
        if input_var is None:
            with tf.variable_scope(name):
                input_var = tf.placeholder(tf.float32, shape=shape, name="input")
        self.input_var = input_var

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape


class MergeLayer(Layer):
    def __init__(self, incomings, name):
        self.input_shapes = [incoming if isinstance(incoming, tuple)
                             else incoming.output_shape
                             for incoming in incomings]
        self.input_layers = [None if isinstance(incoming, tuple)
                             else incoming
                             for incoming in incomings]
        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []

    @Layer.output_shape.getter
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shapes)
        if any(isinstance(s, (tf.Variable, tf.Tensor)) for s in shape):
            raise ValueError("%s returned a symbolic output shape from its "
                             "get_output_shape_for() method: %r. This is not "
                             "allowed; shapes must be tuples of integers for "
                             "fixed-size dimensions and Nones for variable "
                             "dimensions." % (self.__class__.__name__, shape))
        return shape

    def get_output_shape_for(self, input_shapes):
        raise NotImplementedError

    def get_output_for(self, inputs, **kwargs):
        raise NotImplementedError


class ConcatLayer(MergeLayer):
    """
    Concatenates multiple inputs along the specified axis. Inputs should have
    the same shape except for the dimension specified in axis, which can have
    different sizes.
    Parameters
    -----------
    incomings : a list of :class:`Layer` instances or tuples
        The layers feeding into this layer, or expected input shapes
    axis : int
        Axis which inputs are joined over
    """

    def __init__(self, incomings, name, axis=1, **kwargs):
        super(ConcatLayer, self).__init__(incomings, name, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shapes):
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = [next((s for s in sizes if s is not None), None)
                        for sizes in zip(*input_shapes)]

        def match(shape1, shape2):
            return (len(shape1) == len(shape2) and
                    all(i == self.axis or s1 is None or s2 is None or s1 == s2
                        for i, (s1, s2) in enumerate(zip(shape1, shape2))))

        # Check for compatibility with inferred output shape
        if not all(match(shape, output_shape) for shape in input_shapes):
            raise ValueError("Mismatch: input shapes must be the same except "
                             "in the concatenation axis")
        # Infer output shape on concatenation axis and return
        sizes = [input_shape[self.axis] for input_shape in input_shapes]
        concat_size = None if any(s is None for s in sizes) else sum(sizes)
        output_shape[self.axis] = concat_size
        return tuple(output_shape)

    def get_output_for(self, inputs, **kwargs):
        dtypes = [x.dtype.as_numpy_dtype for x in inputs]
        if len(set(dtypes)) > 1:
            # need to convert to common data type
            common_dtype = np.core.numerictypes.find_common_type([], dtypes)
            inputs = [tf.cast(x, common_dtype) for x in inputs]
        return tf.concat(concat_dim=self.axis, values=inputs)


concat = ConcatLayer  # shortcut


def xavier_init(shape, dtype=tf.float32):
    assert len(shape) == 2
    n_inputs, n_outputs = shape
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range, dtype=dtype)(shape)


class ParamLayer(Layer):
    def __init__(self, incoming, num_units, param=tf.zeros_initializer,
                 trainable=True, **kwargs):
        super(ParamLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.param = self.add_param(
            param,
            (num_units,),
            name="param",
            trainable=trainable
        )

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        ndim = input.get_shape().ndims
        reshaped_param = tf.reshape(self.param, (1,) * (ndim - 1) + (self.num_units,))
        tile_arg = tf.concat(0, [tf.shape(input)[:ndim - 1], [1]])
        tiled = tf.tile(reshaped_param, tile_arg)
        return tiled


class DenseLayer(Layer):
    def __init__(self, incoming, name, num_units, nonlinearity=None, W=xavier_init, b=tf.zeros_initializer,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming, name, **kwargs)
        self.nonlinearity = tf.identity if nonlinearity is None else nonlinearity

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.get_shape().ndims > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = tf.reshape(input, tf.pack([tf.shape(input)[0], -1]))

        activation = tf.matmul(input, self.W)
        if self.b is not None:
            activation = activation + tf.expand_dims(self.b, 0)
        return self.nonlinearity(activation)


def get_all_layers(layer, treat_as_input=None):
    """
    :type layer: Layer | list[Layer]
    :rtype: list[Layer]
    """
    # We perform a depth-first search. We add a layer to the result list only
    # after adding all its incoming layers (if any) or when detecting a cycle.
    # We use a LIFO stack to avoid ever running into recursion depth limits.
    try:
        queue = deque(layer)
    except TypeError:
        queue = deque([layer])
    seen = set()
    done = set()
    result = []

    # If treat_as_input is given, we pretend we've already collected all their
    # incoming layers.
    if treat_as_input is not None:
        seen.update(treat_as_input)

    while queue:
        # Peek at the leftmost node in the queue.
        layer = queue[0]
        if layer is None:
            # Some node had an input_layer set to `None`. Just ignore it.
            queue.popleft()
        elif layer not in seen:
            # We haven't seen this node yet: Mark it and queue all incomings
            # to be processed first. If there are no incomings, the node will
            # be appended to the result list in the next iteration.
            seen.add(layer)
            if hasattr(layer, 'input_layers'):
                queue.extendleft(reversed(layer.input_layers))
            elif hasattr(layer, 'input_layer'):
                queue.appendleft(layer.input_layer)
        else:
            # We've been here before: Either we've finished all its incomings,
            # or we've detected a cycle. In both cases, we remove the layer
            # from the queue and append it to the result list.
            queue.popleft()
            if layer not in done:
                result.append(layer)
                done.add(layer)

    return result


def get_output(layer_or_layers, inputs=None, **kwargs):
    # track accepted kwargs used by get_output_for
    accepted_kwargs = {'deterministic'}
    # obtain topological ordering of all layers the output layer(s) depend on
    treat_as_input = inputs.keys() if isinstance(inputs, dict) else []
    all_layers = get_all_layers(layer_or_layers, treat_as_input)
    # initialize layer-to-expression mapping from all input layers
    all_outputs = dict((layer, layer.input_var)
                       for layer in all_layers
                       if isinstance(layer, InputLayer) and
                       layer not in treat_as_input)
    # update layer-to-expression mapping from given input(s), if any
    if isinstance(inputs, dict):
        all_outputs.update((layer, tf.convert_to_tensor(expr))
                           for layer, expr in inputs.items())
    elif inputs is not None:
        if len(all_outputs) > 1:
            raise ValueError("get_output() was called with a single input "
                             "expression on a network with multiple input "
                             "layers. Please call it with a dictionary of "
                             "input expressions instead.")
        for input_layer in all_outputs:
            all_outputs[input_layer] = tf.convert_to_tensor(inputs)
    # update layer-to-expression mapping by propagating the inputs
    for layer in all_layers:
        if layer not in all_outputs:
            try:
                if isinstance(layer, MergeLayer):
                    layer_inputs = [all_outputs[input_layer]
                                    for input_layer in layer.input_layers]
                else:
                    layer_inputs = all_outputs[layer.input_layer]
            except KeyError:
                # one of the input_layer attributes must have been `None`
                raise ValueError("get_output() was called without giving an "
                                 "input expression for the free-floating "
                                 "layer %r. Please call it with a dictionary "
                                 "mapping this layer to an input expression."
                                 % layer)
            all_outputs[layer] = layer.get_output_for(layer_inputs, **kwargs)
            try:
                names, _, _, defaults = getargspec(layer.get_output_for)
            except TypeError:
                # If introspection is not possible, skip it
                pass
            else:
                if defaults is not None:
                    accepted_kwargs |= set(names[-len(defaults):])
            accepted_kwargs |= set(layer.get_output_kwargs)
    unused_kwargs = set(kwargs.keys()) - accepted_kwargs
    if unused_kwargs:
        suggestions = []
        for kwarg in unused_kwargs:
            suggestion = get_close_matches(kwarg, accepted_kwargs)
            if suggestion:
                suggestions.append('%s (perhaps you meant %s)'
                                   % (kwarg, suggestion[0]))
            else:
                suggestions.append(kwarg)
        warn("get_output() was called with unused kwargs:\n\t%s"
             % "\n\t".join(suggestions))
    # return the output(s) of the requested layer(s) only
    try:
        return [all_outputs[layer] for layer in layer_or_layers]
    except TypeError:
        return all_outputs[layer_or_layers]


def unique(l):
    """Filters duplicates of iterable.
    Create a new list from l with duplicate entries removed,
    while preserving the original order.
    Parameters
    ----------
    l : iterable
        Input iterable to filter of duplicates.
    Returns
    -------
    list
        A list of elements of `l` without duplicates and in the same order.
    """
    new_list = []
    seen = set()
    for el in l:
        if el not in seen:
            new_list.append(el)
            seen.add(el)

    return new_list


def get_all_params(layer, **tags):
    """
    :type layer: Layer|list[Layer]
    """
    layers = get_all_layers(layer)
    params = chain.from_iterable(l.get_params(**tags) for l in layers)
    return unique(params)
