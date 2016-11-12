from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.binary_hash import BinaryHash
from rllab.core.network import ConvNetwork
from rllab.misc.overrides import overrides
from rllab.misc import ext
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

class ConvHash(BinaryHash):
    """
    Apply a convolution to the hashed items (supposedly images); optionally, apply SimHash later
    TODO
    - allow modifying the randomization process
    """
    def __init__(self,
            item_shape,
            output_dim,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_sizes=[],
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            bucket_sizes=None,
            second_hash=None,
            binarize_before_second_hash=True,
            parallel=False,
        ):
        assert len(item_shape) == 3 # (channel, width, height)
        self.convnet = ConvNetwork(
            input_shape=item_shape,
            output_dim=output_dim,
            conv_filters=conv_filters,
            conv_filter_sizes=conv_filter_sizes,
            conv_strides=conv_strides,
            conv_pads=conv_pads,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            name="conv_hash_net",
        )
        self._f_output = ext.compile_function(
            [self.convnet.input_layer.input_var],
            L.get_output(self.convnet.output_layer)
        )

        if second_hash is not None:
            dim_key = second_hash.dim_key
        else:
            dim_key = output_dim

        super().__init__(
            dim_key=dim_key,
            bucket_sizes=bucket_sizes,
            parallel=parallel,
        )

        self.item_shape = item_shape
        self.item_dim = np.prod(item_shape)
        self.output_dim = output_dim
        self.second_hash = second_hash
        self.binarize_before_second_hash = binarize_before_second_hash

    def __getstate__(self):
        return super().__getstate__()

    def compute_binary_keys(self, items):
        # remember to vectorize the items before feeding into the network
        if len(items.shape) != 2: # (index, vectorized)
            assert len(items.shape) == 4 # (index, channel, width, height)
            items = items.reshape((items.shape[0],np.prod(self.item_shape)))
        convnet_outputs = self._f_output(items)

        if self.second_hash is not None:
            if self.binarize_before_second_hash:
                second_hash_inputs = np.sign(convnet_outputs)
            else:
                second_hash_inputs = convnet_outputs
            binaries = self.second_hash.compute_binary_keys(second_hash_inputs)
        else:
            binaries = np.sign(convnet_outputs)

        return binaries
