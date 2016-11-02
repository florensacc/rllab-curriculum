from sandbox.haoran.myscripts.myutilities import form_rgb_img_array
import numpy as np
import time
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
        image_shape,
        cell_size,
        n_bin,
        batch_method="vectorized_v2",
        scale=1.0,
    ):
        """
        :param cell_size: width and height of a cell
        :param n_bin: number of discrete bins for each image channel
        """
        self.cell_size = cell_size
        self.n_bin = n_bin
        self.batch_method = batch_method
        self.scale = scale
        self.image_shape = image_shape

        height, width, channel = self.image_shape
        f_height = np.ceil(scale * height / self.cell_size).astype(int)
        f_width = np.ceil(scale * width / self.cell_size).astype(int)
        self.feature_shape = (f_height, f_width, channel)


        if self.batch_method == "vectorized_v1":
            self.construct_pixel_sum_matrix()

    def construct_pixel_sum_matrix(self):
        height,width,channel = self.image_shape
        f_height,f_width,channel = self.feature_shape
        img_len = np.prod(self.image_shape)
        feature_len = np.prod(self.feature_shape)

        self.sum_pixel_mat = np.zeros((img_len, feature_len),dtype=int)
        img_index = 0
        for y in range(height):
            y_cell = np.floor(y / self.cell_size).astype(int)
            for x in range(width):
                x_cell = np.floor(x / self.cell_size).astype(int)
                for c in range(channel):
                    feature_index = y_cell * f_width * channel + x_cell * channel + c
                    self.sum_pixel_mat[img_index,feature_index] = 1
                    img_index += 1

    def get_feature_length(self):
        return np.prod(self.feature_shape)

    def compute_features_nary(self, images):
        """
        Batch version of compute_feature. Should be faster.
        Reshape images to vectors and vectorize all computation.

        :param images: a list of images of shape (height, width, channel)
        """
        t0 = time.time()
        assert len(images.shape) == 4 and images[0].shape == self.image_shape
        batch_size = len(images)

        if abs(self.scale-1.0) > 1e-4:
            images = [cv2.resize(image,dsize=(0,0),fx=self.scale,fy=self.scale) for image in images]

        if self.batch_method == "naive":
            features = np.asarray([self.compute_feature(image) for image in images]).reshape((batch_size,-1))
        elif self.batch_method == "vectorized_v1" or self.batch_method == "vectorized_v2":
            images = np.asarray(images,dtype=int) # summation in uint8 can cause trouble
            batch_size, height, width, channel = images.shape
            feature_height = np.ceil(height / self.cell_size).astype(int)
            feature_width = np.ceil(width / self.cell_size).astype(int)

            if self.batch_method == "vectorized_v1":
                # ordering: channel, col, row
                # regard this as a matrix vector product problem
                images_flat = images.reshape((batch_size,-1))
                sum_colors = images_flat.dot(self.sum_pixel_mat)
                avg_colors = sum_colors / (self.cell_size) ** 2 # ignore incomplete cells

            elif self.batch_method == "vectorized_v2":
                # ordering: row, col, channel
                def compute_features_one_channel(images_c):
                    dx = self.cell_size
                    dy = self.cell_size
                    col_sums = [
                        np.sum(images_c[:,:,x:x+dx],axis=2)
                        for x in range(0,width,dx)
                    ]
                    cell_sums = np.hstack([
                        np.hstack([
                            np.sum(col_sum[:,y:y+dy],axis=1,keepdims=True)
                            for y in range(0,height,dy)
                        ])
                        for col_sum in col_sums
                    ])
                    avg_colors = cell_sums / (dx * dy)
                    return avg_colors
                avg_colors = np.hstack([
                    compute_features_one_channel(images[:,:,:,c])
                    for c in range(channel)
                ])

            bin_size = 255 / self.n_bin
            features = np.minimum(
                np.floor(avg_colors / bin_size).astype(np.uint8),
                self.n_bin-1
            ).reshape(batch_size,-1) # numeric issues may lead to bin_numbers > self.n_bin-1

        else:
            raise NotImplementedError
        # print("batch_method %s elapsed time: %.4f"%(self.batch_method, time.time() - t0))
        return features

    def render_feature_nary(self,feature_nary):
        height,width,channel = self.image_shape
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
        if self.batch_method == "naive":
            feature = feature_nary.reshape((feature_height, feature_width, channel))
        elif self.batch_method == "vectorized_v1":
            feature = feature_nary.reshape((feature_height, feature_width, channel))
        elif self.batch_method == "vectorized_v2":
            feature = feature_nary.reshape((channel,feature_width,feature_height)).transpose((2,1,0))
        else:
            raise NotImplementedError

        bin_size = np.floor(255.0 / self.n_bin)
        for y_cell in range(feature_height):
            for x_cell in range(feature_width):
                y = y_cell * self.cell_size
                x = x_cell * self.cell_size
                dy = min(height - y, self.cell_size)
                dx = min(width - x, self.cell_size)
                for c in range(channel):
                    visual[y:y+dy, x:x+dx,c] = feature[y_cell,x_cell,c] * bin_size

        return visual

    # def compute_feature(self,image):
    #     """
    #     :param image: must be a uint8 RGB image with shape (height, width, channel)
    #     """
    #     assert image.dtype == np.uint8
    #     assert len(image.shape) == 3
    #     height, width, channel = image.shape
    #     feature_height = np.ceil(height / self.cell_size).astype(int)
    #     feature_width = np.ceil(width / self.cell_size).astype(int)
    #     feature = np.zeros((feature_height, feature_width, channel),dtype=np.uint8)
    #
    #     for y_cell in range(feature_height):
    #         for x_cell in range(feature_width):
    #             y = y_cell * self.cell_size
    #             x = x_cell * self.cell_size
    #             dy = min(height - y, self.cell_size)
    #             dx = min(width - x, self.cell_size)
    #             img_cell = image[y: y + dy, x: x + dx,:]
    #             avg_colors = np.sum(img_cell,axis=(0,1)) / (dy * dx)
    #
    #             bin_size = 255 / self.n_bin
    #             bin_numbers = np.minimum(
    #                 np.floor(avg_colors / bin_size).astype(int),
    #                 self.n_bin-1
    #             ) # numeric issues may lead to bin_numbers > self.n_bin-1
    #             feature[y_cell, x_cell] = bin_numbers
    #
    #     return feature
    #
    # def compute_feature_nary(self, image):
    #     feature = self.compute_feature(image)
    #     return feature.ravel()


    def render_feature(self,image):
        """
        Return the same feature image with the same size as the original image
        """
        feature_nary = self.compute_features_nary(np.asarray([image]))[0]
        visual = self.render_feature_nary(feature_nary)
        return visual

import unittest
