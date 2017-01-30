# from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer, Conv2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer, ReshapeLayer
import lasagne.nonlinearities
import lasagne
import lasagne.init as LI
import theano
from sandbox.dave.rllab.core.lasagne_layers import FlattenLayer

# Caffe reference model lasagne implementation
# http://caffe.berkeleyvision.org/
# License: non-commercial use only

# Pretrained weights (233M) can be downloaded from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/caffe_reference.pkl
def wrapped_conv(*args, **kwargs):
    copy = dict(kwargs)
    copy.pop("image_shape", None)
    copy.pop("filter_shape", None)
    assert copy.pop("filter_flip", False)
    input, W, input_shape, get_W_shape = args
    if theano.config.device == 'cpu':
        return theano.tensor.nnet.conv2d(*args, **copy)
    try:
        return theano.sandbox.cuda.dnn.dnn_conv(
            input.astype('float32'),
            W.astype('float32'),
            **copy
        )
    except Exception as e:
        print("falling back to default conv2d")
        return theano.tensor.nnet.conv2d(*args, **kwargs)

class AlexNet(object):
    def __init__(self, input_layer=None):
        net = {}
        # if input_layer is None:
        net['data'] = input_layer
        # else:
            # net['data'] = input_layer

        # conv1
        net['conv1'] = Conv2DLayer(
            net['data'],
            num_filters=96,
            filter_size=(11, 11),
            stride=4,
            name='conv1',
            convolution=wrapped_conv,
        )

        net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2, name='pool1')

        # norm1
        net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],
                                                         n=5,
                                                         alpha=0.0001 / 5.0,
                                                         beta=0.75,
                                                         k=1,
                                                         name='norm1',
                                                         )

        # conv2
        # The caffe reference model uses a parameter called group.
        # This parameter splits input to the convolutional layer.
        # The first half of the filters operate on the first half
        # of the input from the previous layer. Similarly, the
        # second half operate on the second half of the input.
        #
        # Lasagne does not have this group parameter, but we can
        # do it ourselves.
        #
        # see https://github.com/BVLC/caffe/issues/778
        # also see https://code.google.com/p/cuda-convnet/wiki/LayerParams

        # before conv2 split the data
        net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1, name='conv2_data1')
        net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48, 96), axis=1, name='conv2_data2')

        # now do the convolutions
        net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                         num_filters=128,
                                         filter_size=(5, 5),
                                         pad=2,
                                         name='conv2_part1',
                                         convolution=wrapped_conv,
                                         )
        net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                         num_filters=128,
                                         filter_size=(5, 5),
                                         pad=2,
                                         name='conv2_part2',
                                         convolution=wrapped_conv,
                                         )

        # now combine
        net['conv2'] = concat((net['conv2_part1'], net['conv2_part2']), axis=1, name='conv2')

        # pool2
        net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride=2, name='pool2')

        # norm2
        net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],
                                                         n=5,
                                                         alpha=0.0001 / 5.0,
                                                         beta=0.75,
                                                         k=1,
                                                         name='norm2')

        # names = ['conv1', 'conv2_part1', 'conv2_part2', 'conv3', 'conv4_part1', 'conv4_part2', 'conv5_part1', 'conv5_part2', 'fc6', 'fc7', 'fc8']
        # conv3
        # no group
        net['conv3'] = Conv2DLayer(net['norm2'],
                                   num_filters=384,
                                   filter_size=(3, 3),
                                   pad=1,
                                   name='conv3',
                                   convolution=wrapped_conv,
                                   )

        # conv4
        # group = 2
        net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1, name='conv4_data1')
        net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192, 384), axis=1, name='conv4_data2')
        net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],
                                         num_filters=192,
                                         filter_size=(3, 3),
                                         pad=1,
                                         name='conv4_part1',
                                         convolution=wrapped_conv,
                                         )
        net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],
                                         num_filters=192,
                                         filter_size=(3, 3),
                                         pad=1,
                                         name='conv4_part2',
                                         convolution=wrapped_conv,
                                         )
        net['conv4'] = concat((net['conv4_part1'], net['conv4_part2']), axis=1, name='conv4')

        # conv5
        # group 2
        net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1, name='conv5_data1')
        net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192, 384), axis=1, name='conv5_data2')
        net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],
                                         num_filters=128,
                                         filter_size=(3, 3),
                                         pad=1,

                                         name='conv5_part1',
                                         convolution=wrapped_conv,
                                         )
        net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],
                                         num_filters=128,
                                         filter_size=(3, 3),
                                         pad=1,
                                         name='conv5_part2',
                                         convolution=wrapped_conv,
                                         )
        net['conv5'] = concat((net['conv5_part1'], net['conv5_part2']), axis=1, name='conv5')
        net['output'] = FlattenLayer(net['conv5'])

        # pool 5
        # net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride=2, name='pool5')

        # fc6
        #   net['fc6'] = DenseLayer(
        #     net['pool5'], num_units=4096,
        #     nonlinearity=lasagne.nonlinearities.rectify,
        #     name='fc6',
        #     W=LI.GlorotUniform(),
        #     b=LI.Constant(0.))
        #
        # # fc7
        # net['fc7'] = DenseLayer(
        #     net['fc6'],
        #     num_units=4096,
        #     nonlinearity=lasagne.nonlinearities.rectify,
        #     name='fc7',
        #     W=LI.GlorotUniform(),
        #     b=LI.Constant(0.))
        #
        # # fc8
        # net['fc8'] = DenseLayer(
        #     net['fc7'],
        #     num_units=1000,
        #     nonlinearity=lasagne.nonlinearities.softmax,
        #     name='fc8',
        #     W=LI.GlorotUniform(),
        #     b=LI.Constant(0.))

        # net['fc8'] = DenseLayer(
        #     net['pool5'], num_units=96,
        #     nonlinearity=lasagne.nonlinearities.rectify,
        #     name='fc8',
        #     W=LI.GlorotUniform(),
        #     b=LI.Constant(0.))

        self._l_in = net['data']
        self._layers = net
        self._l_out = net['output']
        self._conv_out = net['output']

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def conv_output_layer(self):
        return self._conv_out

    @property
    def layers(self):
        return self._layers


class VanillaConvNet(object):
    def __init__(self, input_layer):
        net = {}
        # if input_layer is None:
        net['data'] = input_layer
        # else:
            # net['data'] = input_layer

        # conv1
        net['conv1'] = Conv2DLayer(
            net['data'],
            num_filters=96,
            filter_size=(11, 11),
            stride=4,
            name='conv1',
            convolution=wrapped_conv,
        )

        net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2, name='pool1')

        # norm1
        net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],
                                                         n=5,
                                                         alpha=0.0001 / 5.0,
                                                         beta=0.75,
                                                         k=1,
                                                         name='norm1',
                                                         )

        net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1, name='conv2_data1')
        net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48, 96), axis=1, name='conv2_data2')

        # now do the convolutions
        net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                         num_filters=128,
                                         filter_size=(5, 5),
                                         pad=2,
                                         name='conv2_part1',
                                         convolution=wrapped_conv,
                                         )
        net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                         num_filters=128,
                                         filter_size=(5, 5),
                                         pad=2,
                                         name='conv2_part2',
                                         convolution=wrapped_conv,
                                         )

        # now combine
        net['conv2'] = concat((net['conv2_part1'], net['conv2_part2']), axis=1, name='conv2')

        # pool2
        net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride=2, name='pool2')

        net['fc3'] = DenseLayer(
            net['pool2'], num_units=64,
            nonlinearity=lasagne.nonlinearities.rectify,
            name='fc2',
            W=LI.GlorotUniform(),
            b=LI.Constant(0.))
        #
        # net['fc4'] = DenseLayer(
        #     net['fc3'], num_units=10,
        #     nonlinearity=lasagne.nonlinearities.rectify,
        #     name='fc4',
        #     W=LI.GlorotUniform(),
        #     b=LI.Constant(0.))

        self._l_in = net['data']
        self._layers = net
        self._l_out = net['fc3']
        self._conv_out = net['pool2']

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def conv_output_layer(self):
        return self._conv_out

    @property
    def layers(self):
        return self._layers