# encoding: utf-8
import numpy as np
import lasagne.layers as L
from lasagne import init
import theano as T
import lasagne.nonlinearities as nl
import theano.tensor as TT

def set_layer_param_tags(layer, params=None, **tags):
    """
    If params is None, update tags of all parameters, else only update tags of parameters in params.
    """
    for param, param_tags in layer.params.items():
        if params is None or param in params:
            for tag, value in tags.items():
                if value:
                    param_tags.add(tag)
                else:
                    param_tags.discard(tag)

class CropLayer(L.Layer):
    def __init__(self, l_incoming, start_index=None, end_index=None, name=None):
        super(CropLayer, self).__init__(l_incoming, name)
        self.start_index = start_index
        self.end_index = end_index

    def get_output_shape_for(self, input_shape):
        n_batch = input_shape[0]  # out of the obs_robot_var, the first dim is the batch size
        start = 0
        end = input_shape[1]
        if self.start_index:
            start = self.start_index
        if self.end_index:
            end = self.end_index
        new_length = end - start
        return n_batch, new_length  # this automatically creates a tuple

    def get_output_for(self, all_obs_var, **kwargs):
        return all_obs_var[:, self.start_index:self.end_index]


class ConstOutputLayer(L.Layer):
    def __init__(self, output_var=None, incoming=None, name=None, input_var=None, input_shape=None):
        super(ConstOutputLayer, self).__init__(incoming, name)
        self.output_var = output_var

    def get_output_shape_for(self, input_shape):
        n_batch = input_shape[0]  # the batch size
        single_output_shape = self.output_var.get_value().shape
        return (n_batch,) + single_output_shape  # this is supposed to create a tuple

    def get_output_for(self, all_obs_var, **kwargs):
        n_batch = all_obs_var.shape[0]
        out = TT.tile(self.output_var, (n_batch, 1))
        return out

class ElemwiseDiffLayer(L.MergeLayer):
    def get_output_shape_for(self, input_shapes):
        if any(shape != input_shapes[0] for shape in input_shapes):
            raise ValueError("Mismatch: not all input shapes are the same")
        return input_shapes[0]

    def get_output_for(self, inputs):
        output = None
        for input in inputs:
            if output is not None:
                output = output - input
            else:
                output = input
        return output

class FlattenLayer(L.Layer):
    """
    A layer that flattens its input. The leading ``outdim-1`` dimensions of
    the output will have the same shape as the input. The remaining dimensions
    are collapsed into the last dimension.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    outdim : int
        The number of dimensions in the output.
    See Also
    --------
    flatten  : Shortcut
    """
    def __init__(self, incoming, outdim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.outdim = outdim

        if outdim < 1:
            raise ValueError('Dim must be >0, was %i', outdim)

    def get_output_shape_for(self, input_shape):
        to_flatten = input_shape[self.outdim - 1:]

        if any(s is None for s in to_flatten):
            flattened = None
        else:
            flattened = int(np.prod(to_flatten))

        return input_shape[:self.outdim - 1] + (flattened,)

    def get_output_for(self, input, **kwargs):
        return input.flatten(self.outdim)

flatten = FlattenLayer  # shortcut

class CompositionLayer(L.Layer):
    def __init__(self, incoming, layers=None, name=None):
        super(CompositionLayer, self).__init__(incoming, name=name)
        self.layers = []
        for layer in layers or []:
            self.add_layer(layer)

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params.update(layer.params)
        return layer

    def get_output_shape_for(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            shape = layer.get_output_shape_for(shape)
        return shape

    def get_output_for(self, input, **kwargs):
        output = input
        for layer in self.layers:
            output = layer.get_output_for(output, **kwargs)
        return output

    def get_param_kwargs(self, **tags):
        params = self.get_params(**tags)
        return dict([(self.param_keys[param], param) for param in params])

class DilatedVggEncodingLayer(CompositionLayer):
    def __init__(self, incoming, num_filters, filter_size=3, dilation=(2, 2),
                 conv1_W=init.GlorotUniform(), conv1_b=init.Constant(0.),
                 bn1_beta=init.Constant(0.), bn1_gamma=init.Constant(1.),
                 bn1_mean=init.Constant(0.), bn1_inv_std=init.Constant(1.),
                 conv2_W=init.GlorotUniform(), conv2_b=init.Constant(0.),
                 bn2_beta=init.Constant(0.), bn2_gamma=init.Constant(1.),
                 bn2_mean=init.Constant(0.), bn2_inv_std=init.Constant(1.),
                 batch_norm=False, name=None,
                 **tags):
        super(DilatedVggEncodingLayer, self).__init__(incoming, name=name)
        layer = self.l_conv1 = self.add_layer(
            L.Conv2DLayer(incoming, num_filters, filter_size=filter_size, stride=1, pad='same', nonlinearity=None,
                          W=conv1_W,
                          b=conv1_b,
                          name='%s.%s' % (name, 'conv1') if name is not None else None))
        if batch_norm:
            layer = self.l_bn1 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn1_beta,
                                 gamma=bn1_gamma,
                                 mean=bn1_mean,
                                 inv_std=bn1_inv_std,
                                 name='%s.%s' % (name, 'bn1') if name is not None else None))
        else:
            self.l_bn1 = None
        layer = self.l_relu1 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu1') if name is not None else None))

        # layer = self.l_pad2 = self.add_layer(
        #     L.PadLayer(layer, (filter_size - 1) * dilation // 2))  # 'same' padding
        # layer = self.l_conv2 = self.add_layer(
        #     L.DilatedConv2DLayer(layer, num_filters, filter_size=filter_size, dilation=dilation, nonlinearity=None,
        #                          W=conv2_W,
        #                          b=conv2_b,
        #                          name='%s.%s' % (name, 'conv2') if name is not None else None))
        #TODO: PAD ERROR (It doesn't exists!!)
        layer = self.l_conv2 = self.add_layer(
            L.DilatedConv2DLayer(layer, num_filters, filter_size=filter_size, dilation=dilation, pad='same', nonlinearity=None,
                          W=conv2_W,
                          b=conv2_b,
                          name='%s.%s' % (name, 'conv2') if name is not None else None))
        if batch_norm:
            layer = self.l_bn2 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn2_beta,
                                 gamma=bn2_gamma,
                                 mean=bn2_mean,
                                 inv_std=bn2_inv_std,
                                 name='%s.%s' % (name, 'bn2') if name is not None else None))
        else:
            self.l_bn2 = None
        self.l_relu2 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu2') if name is not None else None))

        for tag in tags.keys():
            if not isinstance(tag, str):
                raise ValueError("tag should be a string, %s given" % type(tag))
        tags['encoding'] = tags.get('encoding', True)
        set_layer_param_tags(self, **tags)

        self.param_keys = dict()
        for layer, base_name in [(self.l_conv1, 'conv1'), (self.l_conv2, 'conv2')]:
            self.param_keys.update({
                layer.W: '%s_W' % base_name,
                layer.b: '%s_b' % base_name,
            })
        for layer, base_name in [(self.l_bn1, 'bn1'), (self.l_bn2, 'bn2')]:
            if layer is not None:
                self.param_keys.update({
                    layer.beta: '%s_beta' % base_name,
                    layer.gamma: '%s_gamma' % base_name,
                    layer.mean: '%s_mean' % base_name,
                    layer.inv_std: '%s_inv_std' % base_name
                })


class DilatedVggEncoding3Layer(CompositionLayer):
    def __init__(self, incoming, num_filters, filter_size=3, dilation=(2, 2),
                 conv1_W=init.GlorotUniform(), conv1_b=init.Constant(0.),
                 bn1_beta=init.Constant(0.), bn1_gamma=init.Constant(1.),
                 bn1_mean=init.Constant(0.), bn1_inv_std=init.Constant(1.),
                 conv2_W=init.GlorotUniform(), conv2_b=init.Constant(0.),
                 bn2_beta=init.Constant(0.), bn2_gamma=init.Constant(1.),
                 bn2_mean=init.Constant(0.), bn2_inv_std=init.Constant(1.),
                 conv3_W=init.GlorotUniform(), conv3_b=init.Constant(0.),
                 bn3_beta=init.Constant(0.), bn3_gamma=init.Constant(1.),
                 bn3_mean=init.Constant(0.), bn3_inv_std=init.Constant(1.),
                 batch_norm=False, name=None,
                 **tags):
        super(DilatedVggEncoding3Layer, self).__init__(incoming, name=name)
        layer = self.l_conv1 = self.add_layer(
            L.Conv2DLayer(incoming, num_filters, filter_size=filter_size, stride=1, pad='same',
                          nonlinearity=None,
                          W=conv1_W,
                          b=conv1_b,
                          name='%s.%s' % (name, 'conv1') if name is not None else None))
        if batch_norm:
            layer = self.l_bn1 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn1_beta,
                                 gamma=bn1_gamma,
                                 mean=bn1_mean,
                                 inv_std=bn1_inv_std,
                                 name='%s.%s' % (name, 'bn1') if name is not None else None))
        else:
            self.l_bn1 = None
        layer = self.l_relu1 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu1') if name is not None else None))

        layer = self.l_conv2 = self.add_layer(
            L.Conv2DLayer(layer, num_filters, filter_size=filter_size, stride=1, pad='same',
                          nonlinearity=None,
                          W=conv2_W,
                          b=conv2_b,
                          name='%s.%s' % (name, 'conv2') if name is not None else None))
        if batch_norm:
            layer = self.l_bn2 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn2_beta,
                                 gamma=bn2_gamma,
                                 mean=bn2_mean,
                                 inv_std=bn2_inv_std,
                                 name='%s.%s' % (name, 'bn2') if name is not None else None))
        else:
            self.l_bn2 = None
        layer = self.l_relu2 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu2') if name is not None else None))

        # layer = self.l_pad3 = self.add_layer(
        #     L.PadLayer(layer, (filter_size - 1) * dilation // 2))  # 'same' padding
        # layer = self.l_conv3 = self.add_layer(
        #     L.DilatedConv2DLayer(layer, num_filters, filter_size=filter_size, dilation=dilation, nonlinearity=None,
        #                          W=conv3_W,
        #                          b=conv3_b,
        #                          name='%s.%s' % (name, 'conv3') if name is not None else None))
        layer = self.l_conv3 = self.add_layer(
            L.Conv2DLayer(layer, num_filters, filter_size=filter_size, filter_dilation=dilation, pad='same',
                          nonlinearity=None,
                          W=conv3_W,
                          b=conv3_b,
                          name='%s.%s' % (name, 'conv3') if name is not None else None))
        if batch_norm:
            layer = self.l_bn3 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn3_beta,
                                 gamma=bn3_gamma,
                                 mean=bn3_mean,
                                 inv_std=bn3_inv_std,
                                 name='%s.%s' % (name, 'bn3') if name is not None else None))
        else:
            self.l_bn3 = None
        self.l_relu3 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu3') if name is not None else None))

        for tag in tags.keys():
            if not isinstance(tag, str):
                raise ValueError("tag should be a string, %s given" % type(tag))
        tags['encoding'] = tags.get('encoding', True)
        set_layer_param_tags(self, **tags)

        self.param_keys = dict()
        for layer, base_name in [(self.l_conv1, 'conv1'), (self.l_conv2, 'conv2'), (self.l_conv3, 'conv3')]:
            self.param_keys.update({
                layer.W: '%s_W' % base_name,
                layer.b: '%s_b' % base_name,
            })
        for layer, base_name in [(self.l_bn1, 'bn1'), (self.l_bn2, 'bn2'), (self.l_bn3, 'bn3')]:
            if layer is not None:
                self.param_keys.update({
                    layer.beta: '%s_beta' % base_name,
                    layer.gamma: '%s_gamma' % base_name,
                    layer.mean: '%s_mean' % base_name,
                    layer.inv_std: '%s_inv_std' % base_name
                })
