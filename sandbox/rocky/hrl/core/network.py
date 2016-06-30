from __future__ import print_function
from __future__ import absolute_import
import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as LN
import itertools
import numpy as np
from rllab.core.network import wrapped_conv
from rllab.core.serializable import Serializable
import theano.tensor as TT
import itertools


class MergeMLP(Serializable):
    """
    A more general version of the usual multi-layer perceptron. It first split the input layer into multiple
    branches, specified by the `branch_dim` parameter, passing each one of them into several layers,
    and then concatenate them together followed by some number of layers after that.
    """

    def __init__(self, input_shape, branch_dims, output_dim, branch_hidden_sizes, joint_hidden_sizes,
                 hidden_nonlinearity, output_nonlinearity, hidden_W_init=LI.GlorotUniform(),
                 hidden_b_init=LI.Constant(0.), output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 name=None, input_var=None):

        Serializable.quick_init(self, locals())

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        input_dim = np.prod(input_shape)
        assert input_dim == np.sum(branch_dims)
        l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        if len(input_shape) > 1:
            l_reshaped_in = L.ReshapeLayer(l_in, (input_dim,))
        else:
            l_reshaped_in = l_in

        l_branch_hids = []

        branch_offset = 0

        for branch_idx, branch_dim, hidden_sizes in zip(itertools.count(), branch_dims, branch_hidden_sizes):
            l_hid = L.SliceLayer(l_reshaped_in, slice(branch_offset, branch_offset + branch_dim), axis=-1)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="%sbranch_%d_hidden_%d" % (prefix, branch_idx, idx),
                    W=hidden_W_init,
                    b=hidden_b_init,
                )
            branch_offset += branch_dim
            l_branch_hids.append(l_hid)

        l_joint_hid = L.concat(l_branch_hids)

        for idx, hidden_size in enumerate(joint_hidden_sizes):
            l_joint_hid = L.DenseLayer(
                l_joint_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%sjoint_hidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )

        l_out = L.DenseLayer(
            l_joint_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        # self._layers.append(l_out)
        self._l_in = l_in
        self._l_out = l_out
        self._input_var = l_in.input_var
        self._output = L.get_output(l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._input_var

    @property
    def output(self):
        return self._output


class ConvMergeNetwork(Serializable):
    """
    This network allows the input to consist of a convolution-friendly component, plus a non-convolution-friendly
    component. These two components will be concatenated in the fully connected layers. There can also be a list of
    optional layers for the non-convolution-friendly component alone.


    The input to the network should be a matrix where each row is a single input entry, with both the aforementioned
    components flattened out and then concatenated together
    """

    def __init__(self, input_shape, extra_input_shape, output_dim, hidden_sizes,
                 conv_filters, conv_filter_sizes, conv_strides, conv_pads,
                 extra_hidden_sizes=None,
                 hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 hidden_nonlinearity=LN.rectify,
                 output_nonlinearity=None,
                 name=None, input_var=None, input_layer=None):
        Serializable.quick_init(self, locals())

        if extra_hidden_sizes is None:
            extra_hidden_sizes = []

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        input_flat_dim = np.prod(input_shape)
        extra_input_flat_dim = np.prod(extra_input_shape)
        total_input_flat_dim = input_flat_dim + extra_input_flat_dim

        if input_layer is None:
            l_in = L.InputLayer(shape=(None, total_input_flat_dim), input_var=input_var)
        else:
            l_in = input_layer

        l_conv_in = L.reshape(L.SliceLayer(l_in, indices=slice(input_flat_dim)), ([0],) + input_shape)
        l_extra_in = L.reshape(L.SliceLayer(l_in, indices=slice(input_flat_dim, None)), ([0],) + extra_input_shape)

        l_conv_hid = l_conv_in
        for idx, conv_filter, filter_size, stride, pad in itertools.izip(
                xrange(len(conv_filters)),
                conv_filters,
                conv_filter_sizes,
                conv_strides,
                conv_pads,
        ):
            l_conv_hid = L.Conv2DLayer(
                l_conv_hid,
                num_filters=conv_filter,
                filter_size=filter_size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=hidden_nonlinearity,
                name="%sconv_hidden_%d" % (prefix, idx),
                convolution=wrapped_conv,
            )

        l_extra_hid = l_extra_in
        for idx, hidden_size in enumerate(extra_hidden_sizes):
            l_extra_hid = L.DenseLayer(
                l_extra_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%sextra_hidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )

        l_joint_hid = L.concat([L.flatten(l_conv_hid), l_extra_hid])

        for idx, hidden_size in enumerate(hidden_sizes):
            l_joint_hid = L.DenseLayer(
                l_joint_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%sjoint_hidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
        l_out = L.DenseLayer(
            l_joint_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        self._l_in = l_in
        self._l_out = l_out

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var


class ActionConditionedConvDeconvNetwork(Serializable):
    """
    This network is modeled after [1]. It assumes that the input is an image and the output is an image of the shape.
    In addition it also receives a vector-shaped action as input. The image input is first passed through several
    convolutional layers and maybe fully connected layers, and adds an action-conditioned offset,
    and a symmetric set of layers is appleid (e.g. transposed fc layers & deconv layers) to get the
    predicted next image.

    Currently, it does NOT share weights between the conv / deconv layers.
    """

    def __init__(self, image_shape, action_dim, embedding_dim,
                 conv_filters, conv_filter_sizes, conv_strides, conv_pads,
                 extra_hidden_sizes=None,
                 W_init=LI.GlorotUniform(), b_init=LI.Constant(0.),
                 hidden_nonlinearity=LN.rectify,
                 output_nonlinearity=None,
                 name=None, input_var=None):
        """
        :param image_shape: Shape of the image. Should be a 3-tuple
        :param action_dim: Dimension of the action
        :param embedding_dim: Dimension of the shared embedding space for actions and images, where an elementwise
        multiplication is applied to them
        :param conv_filters: List of numbers of convolution filters for each conv layer
        :param conv_filter_sizes: List of sizes of convolution filders for each conv layer
        :param conv_strides: List of strides of convolution filders for each conv layer
        :param conv_pads: List of paddings of convolution filders for each conv layer
        :param hidden_W_init: Initializer for W
        :param hidden_b_init: Initializer for b
        :param hidden_nonlinearity: nonlinearity for hidden layers
        :param output_nonlinearity: nonlinearity for output layer
        :param name: name of the network
        :param input_var: symbolic variable for the input
        :return:
        """
        Serializable.quick_init(self, locals())

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        image_flat_dim = np.prod(image_shape)
        input_flat_dim = image_flat_dim + action_dim

        l_in = L.InputLayer(shape=(None, input_flat_dim), input_var=input_var)

        l_conv_in = L.reshape(L.SliceLayer(l_in, indices=slice(image_flat_dim)), ([0],) + image_shape)
        l_action_in = L.reshape(L.SliceLayer(l_in, indices=slice(image_flat_dim, None)), ([0], action_dim))

        l_conv_hid = l_conv_in
        l_conv_hids = []
        for idx, conv_filter, filter_size, stride, pad in itertools.izip(
                xrange(len(conv_filters)),
                conv_filters,
                conv_filter_sizes,
                conv_strides,
                conv_pads,
        ):
            l_conv_hids.append(l_conv_hid)
            l_conv_hid = L.Conv2DLayer(
                l_conv_hid,
                num_filters=conv_filter,
                filter_size=filter_size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=hidden_nonlinearity,
                name="%sconv_hidden_%d" % (prefix, idx),
                convolution=wrapped_conv,
                W=W_init,
                b=b_init,
            )

        l_action_embedding = L.DenseLayer(
            l_action_in,
            num_units=embedding_dim,
            nonlinearity=LN.identity,
            name="%saction_embedding" % prefix,
            W=W_init,
            b=None,
        )

        l_conv_embedding = L.DenseLayer(
            L.flatten(l_conv_hid),
            num_units=embedding_dim,
            nonlinearity=LN.identity,
            name="%sconv_embedding" % prefix,
            W=W_init,
            b=None,
        )

        l_mul = L.ElemwiseMergeLayer(
            [l_action_embedding, l_conv_embedding],
            merge_function=TT.mul,
        )

        l_embed_to_hid = L.ElemwiseSumLayer(
            [
                L.ReshapeLayer(
                    L.DenseLayer(
                        l_mul,
                        num_units=np.prod(l_conv_hid.output_shape[1:]),
                        nonlinearity=hidden_nonlinearity,
                        name="%sembed_to_hid" % prefix,
                        W=W_init,
                        b=b_init,
                    ),
                    shape=([0],) + l_conv_hid.output_shape[1:],
                ),
                l_conv_hid,
            ]
        )

        l_deconv_hid = l_embed_to_hid

        for idx, conv_filter, filter_size, stride, pad in zip(
                xrange(len(conv_filters)),
                conv_filters,
                conv_filter_sizes,
                conv_strides,
                conv_pads,
        )[::-1]:
            forward_layer = L.Conv2DLayer(
                l_conv_hids[idx],
                num_filters=conv_filter,
                filter_size=filter_size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=hidden_nonlinearity,
                convolution=wrapped_conv,
                name="%sstub_conv_%s" % (prefix, idx),
                W=W_init,
                b=b_init,
            )

            l_deconv_hid = L.InverseLayer(l_deconv_hid, forward_layer)

        l_out = L.flatten(l_deconv_hid)

        self._l_in = l_in
        self._l_out = l_out
        self._input_var = l_in.input_var

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var
