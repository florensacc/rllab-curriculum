"""
Allows more tuning of convnet architecture
"""


from rllab.core.network import ConvNetwork
from rllab.misc.ext import compile_function
import lasagne.layers as L
import numpy as np
import cv2


class SpatialSoftmaxFeatureExtractor(object):
    """
    Implementation of feature extractor from https://arxiv.org/abs/1504.00702 with a random convnet.
    """

    def __init__(
        self,
        image_shape,
        conv_filters, conv_filter_sizes, conv_strides, conv_pads,
        slice_size=100,
    ):
        self.image_shape = image_shape
        self.output_dim = conv_filters[-1] * 2

        self.conv = ConvNetwork(
            input_shape=image_shape,
            output_dim=self.output_dim,
            hidden_sizes=(),  # Fully connected layers; don't want any
            conv_filters=conv_filters,
            conv_filter_sizes=conv_filter_sizes,
            conv_strides=conv_strides,
            conv_pads=conv_pads,
            name="RandomConvolutions"
        )
        image_var = self.conv.input_layer.input_var
        conv_outputs = self.conv.conv_output_layer
        conv_outputs_var = L.get_output(conv_outputs, {self.conv.input_layer: image_var})
        self._f_conv_outputs = compile_function([image_var], conv_outputs_var)
        self.slice_size = slice_size

        self.colors = None

    def _spatial_softmax_expectations(self, conv_output):
        """
        Returns features in the form
        [[exp_x_1 ... exp_x_c]
         [exp_y_1 ... exp_y_c]]
        per item (so shape (n, c, 2)) where c is the number of conv channels.
        """
        n, c, w, h = conv_output.shape
        unnormalized_probs = np.exp(conv_output)
        denom = unnormalized_probs.sum(axis=-1).sum(axis=-1)  # sum per channel
        probs = unnormalized_probs / denom.reshape(n, c, 1, 1)
        xs = np.arange(w).reshape(1, 1, -1, 1)
        ys = np.arange(h).reshape(1, 1, 1, -1)
        exp_x = (probs * xs).sum(axis=-1).sum(axis=-1)
        exp_y = (probs * ys).sum(axis=-1).sum(axis=-1)
        return np.concatenate([exp_x.reshape(n, -1, 1), exp_y.reshape(n, -1, 1)], axis=2)

    def get_spatial_softmax_expectations(self, image):
        conv_output = self.get_conv_output(image)
        return self._spatial_softmax_expectations(conv_output)

    def get_conv_output(self, image):
        slices = []
        for i in range(0, len(image), self.slice_size):
            curr_slice = image[i:i+self.slice_size]
            slices.append(self._f_conv_outputs(curr_slice))
        return np.concatenate(slices, axis=0)

    def get_features(self, image):
        ss_exps = self.get_spatial_softmax_expectations(image)
        return ss_exps.reshape(ss_exps.shape[0], -1)

    @property
    def feature_dim(self):
        return self.output_dim

    def render_feature_nary(self, feature_nary):
        self.scale = 0.5
        self.cell_size = 10
        self.n_bin = 20

        channel, width, height = self.image_shape
        _, output_dim = feature_nary.shape
        feature_height = output_dim // 4
        feature_width = 4

        visual = np.zeros(
            shape=(
                np.round(height*self.scale*2).astype(int),
                np.round(width*self.scale/2).astype(int),
                channel
            ),
            dtype=np.uint8
        )
        feature = feature_nary.reshape((feature_height, feature_width, 1))
        # bin_size = np.floor(255.0 / self.n_bin)
        for y_cell in range(feature_height):
            for x_cell in range(feature_width):
                y = y_cell * self.cell_size
                x = x_cell * self.cell_size
                dy = min(height - y, self.cell_size)
                dx = min(width - x, self.cell_size)
                for c in range(1):
                    visual[y:y+dy, x:x+dx, c] = feature[y_cell, x_cell, c] * 10

        return visual

    def render_feature(self, image):
        """
        Return the same feature image with the same size as the original image
        """
        return self.render_feature_spatial(image)
        image = self.subsample_rgb(image)
        features = self.get_features(np.asarray([image]).reshape(1, -1))
        visual = self.render_feature_nary(features)
        return visual

    def subsample_rgb(self, image):
        full_height, full_width, channels = image.shape
        channels, img_height, img_width = self.image_shape
        subsampled_rgb_images = np.zeros((img_height, img_width, channels))
        for channel in range(channels):
            subsampled_rgb_images[:, :, channel] = cv2.resize(
                image[:, :, channel],
                (img_width, img_height))
        image = subsampled_rgb_images
        return image

    def render_feature_spatial(self, image):
        image = self.subsample_rgb(image)
        feature = self.get_spatial_softmax_expectations(np.asarray([image]).reshape(1, -1))
        self.scale = 1
        self.cell_size = 5

        channel, width, height = self.image_shape
        _, num_feature_channels, _ = feature.shape
        feature_height = np.ceil(height * self.scale / self.cell_size).astype(int)
        feature_width = np.ceil(width * self.scale / self.cell_size).astype(int)

        visual = np.zeros(
            shape=(
                np.round(height*self.scale).astype(int),
                np.round(width*self.scale).astype(int),
                channel
            ),
            dtype=np.uint8
        )

        if self.colors is None:
            self.colors = np.random.randint(255, size=(num_feature_channels, channel))

        for y_cell in range(feature_height):
            for x_cell in range(feature_width):
                y = y_cell * self.cell_size
                x = x_cell * self.cell_size
                dy = min(height - y, self.cell_size)
                dx = min(width - x, self.cell_size)
                cell = Cell(x, y, dx, dy)
                for feature_channel in range(num_feature_channels):
                    exp_location = feature[:, feature_channel].flatten()
                    if exp_location in cell:
                        visual[y:y+dy, x:x+dx, :] = self.colors[feature_channel, :]

        return visual


class Cell:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def __contains__(self, item):
        x, y = item
        in_x = x >= self.x and x <= self.x + self.dx
        in_y = y >= self.y and y <= self.y + self.dy
        return in_x and in_y
