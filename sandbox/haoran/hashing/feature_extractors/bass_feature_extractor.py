from sandbox.haoran.myscripts.myutilities import form_rgb_img_array
import numpy as np
import cv2

class BassFeatureExtractor(object):
    """
    Loosely inspired by the BASS feature extractor from Bellemare, 2012.
    Divide the game screen into cells, discritize each image channel.
    Problems:
    - the colors could have range much smaller than 255, in which naive discretization will make many color blocks zero.
    """
    def __init__(
        self,
        cell_size,
        n_bin,
    ):
        """
        :param cell_size: width and height of a cell
        :param n_bin: number of discrete bins for each image channel
        """
        self.cell_size = cell_size
        self.n_bin = n_bin

    def get_feature_shape(self,image_shape):
        """
        :param image_shape: (height, width)
        """
        height, width, channel = image_shape
        f_height = np.ceil(height / self.cell_size).astype(int)
        f_width = np.ceil(width / self.cell_size).astype(int)
        feature_shape = (f_height, f_width, channel)

        return feature_shape

    def get_feature_length(self, image_shape):
        return np.prod(self.get_feature_shape(image_shape))

    def compute_feature(self,image):
        assert image.dtype == np.uint8
        assert len(image.shape) == 3
        height, width, channel = image.shape
        feature_height = np.ceil(height / self.cell_size).astype(int)
        feature_width = np.ceil(width / self.cell_size).astype(int)
        feature = np.zeros((feature_height, feature_width, channel),dtype=np.uint8)

        for y_cell in range(feature_height):
            for x_cell in range(feature_width):
                y_start = y_cell * self.cell_size
                x_start = x_cell * self.cell_size
                dy = min(height - y_start, self.cell_size)
                dx = min(width - x_start, self.cell_size)
                img_cell = image[y_start: y_start + dy, x_start: x_start + dx,:]
                avg_colors = np.sum(img_cell,axis=(0,1)) / (dy * dx)

                bin_size = 255 / self.n_bin
                bin_numbers = np.minimum(
                    np.floor(avg_colors / bin_size).astype(int),
                    self.n_bin-1
                ) # numeric issues may lead to bin_numbers > self.n_bin-1
                feature[y_cell, x_cell] = bin_numbers

        return feature

    def compute_feature_nary(self, image):
        feature = self.compute_feature(image)
        return feature.ravel()

    def render_feature(self,image):
        """
        Return the same feature image with the same size as the original image
        """
        assert image.dtype == np.uint8
        assert len(image.shape) == 3
        height, width, channel = image.shape
        feature_height = np.ceil(height / self.cell_size).astype(int)
        feature_width = np.ceil(width / self.cell_size).astype(int)
        visual = np.zeros_like(image)

        for y_cell in range(feature_height):
            for x_cell in range(feature_width):
                y_start = y_cell * self.cell_size
                x_start = x_cell * self.cell_size
                dy = min(height - y_start, self.cell_size)
                dx = min(width - x_start, self.cell_size)
                img_cell = image[y_start: y_start + dy, x_start: x_start + dx,:]
                avg_colors = np.sum(img_cell,axis=(0,1)) / (dy * dx)

                bin_size = 255 / self.n_bin
                discretized_colors = np.floor(avg_colors / bin_size) * bin_size
                discretized_colors = np.round(discretized_colors).astype(np.uint8)
                visual[y_start: y_start + dy, x_start: x_start + dx, :] = discretized_colors
        return visual

import unittest
