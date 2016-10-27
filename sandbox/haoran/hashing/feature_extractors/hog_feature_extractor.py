import numpy as np
import cv2

class HOGFeatureExtractor(object):
    """
    Extract histogram of oriented gradients.
    Copied from https://github.com/vlfeat/vlfeat/blob/master/vl/hog.c#L596-L724
    """
    def __init__(
        self,
        num_orientations,
        cell_size,
        oriented=True,
        variant="vanilla",
        bilinear=False,
        transpose=False,
        contribute_to_single_cell=False,
    ):
        """
        @param cell_size: width and height of a cell
        @param num_orientations: number of discrete bins in [0,pi]
        @param transpose: whether the input image is accidentally transposed (currently this param only affects the edge map)
        """
        assert(num_orientations >= 1)

        self.cell_size = cell_size
        self.num_orientations = num_orientations
        self.oriented = oriented
        self.variant = variant
        self.bilinear = bilinear
        self.transpose = transpose
        self.contribute_to_single_cell = contribute_to_single_cell

        orientations = np.asarray(range(self.num_orientations))
        angles = orientations * np.pi / self.num_orientations
        self.orientation_x = np.cos(angles)
        self.orientation_y = np.sin(angles)

        if self.variant == "vanilla":
            self.dimension = self.num_orientations #???
        else:
            raise NotImplementedError

        if oriented:
            self.n_bins = 2 * self.num_orientations
        else:
            self.n_bins = self.num_orientations

        self.feature_shape = None
        self.last_query_image_shape = ()

    def get_feature_shape(self,image_shape):
        """
        Not optimized yet. Hopefully this function is not called frequently.
        @param image_shape: (n_channel, width, height)
        """
        if self.feature_shape is None or self.last_query_image_shape != image_shape:
            features = self.compute_hogs(np.zeros((1,)+image_shape))
            self.feature_shape = features.shape[1:]
        self.last_query_image_shape = image_shape
        return self.feature_shape

    def get_feature_length(self, image_shape):
        return np.prod(self.get_feature_shape(image_shape))


    def compute_hogs(self,images):
        """
        @return: histogram of oriented gradients of shape (batch_size, n_bins, width, height)
        """
        assert len(images.shape) == 4
        hogs_u = self.compute_unnormalized_hogs(images)
        # shape: (batch_size, n_bins, hog_width, hog_height)

        if self.variant == "vanilla":
            # normalize within each cell
            hogs = []
            for hog_u in hogs_u:
                normalizers = np.maximum(1e-6,np.sum(hog_u,axis=0))
                hog = np.asarray([
                    h / normalizers
                    for h in hog_u
                ])
                hogs.append(hog)
            hogs = np.asarray(hogs)
        else:
            raise NotImplementedError

        return hogs

    def compute_unnormalized_hogs(self, images):
        """
        @return unnormalized histogram of oriented gradients; shape: (batch_size, n_bins, hog_width, hog_height)
        """
        cell_size = self.cell_size
        num_orientations = self.num_orientations
        n_bins = self.n_bins
        batch_orientations, batch_orientation_weights, batch_grad_norms = self.compute_discretized_grads(images)
        hogs_u = []
        for image, orientations, orientation_weights, grad_norms in \
            zip(images, batch_orientations,batch_orientation_weights,batch_grad_norms):
            n_channel, width, height = image.shape

            # pixel coordinates (of pixel centers whose grads we computed)
            x_pixel = np.outer(
                np.asarray(range(1,width-1)),
                np.ones(height-2),
            ) + 0.5
            y_pixel = np.outer(
                np.ones(width-2),
                np.asarray(range(1,height-1)),
            ) + 0.5

            # cell coordinates (top-left cell center is (0,0))
            x_cell = x_pixel / cell_size - 0.5
            y_cell = y_pixel / cell_size - 0.5

            # cell indices: range: (0 ~ hog_width-1, 0 ~ hog_height-1)
            x_bin = np.floor(x_cell)
            y_bin = np.floor(y_cell)

            # c_ij: cell center at hog row i, hog column j
            # c_11 ----- c_12
            # ---- pixel ----
            # c_21 ----- c_22
            w_right = x_cell - x_bin
            w_left = 1. - w_right
            w_bottom = y_cell - y_bin
            w_top = 1. - w_bottom
            if hasattr(self, "contribute_to_single_cell") and self.contribute_to_single_cell:
                f = lambda w: (np.sign(w - 0.5)+1)*0.5 # 1 if > 0.5; 0 otherwise
                w_right = f(w_right)
                w_left = f(w_left)
                w_bottom = f(w_bottom)
                w_top = f(w_top)

            hog_width = int((width + cell_size/2) // cell_size)
            hog_height = int((height + cell_size/2) // cell_size)
            hog_flat = np.zeros(
                n_bins * hog_width * hog_height,
                dtype=float
            )

            # hog indices: (bin, x, y)
            indices_topleft = np.clip(
                orientations * hog_width * hog_height + x_bin * hog_height + y_bin,
                0,len(hog_flat)-1,
            ).astype(int).ravel()
            indices_topright = np.clip(
                indices_topleft + hog_height,
                0,len(hog_flat)-1,
            )
            indices_bottomleft = np.clip(
                indices_topleft + 1,
                0,len(hog_flat)-1,
            )
            indices_bottomright = np.clip(
                indices_bottomleft + hog_height,
                0,len(hog_flat)-1,
            )

            contributions = grad_norms * orientation_weights
            add_topleft =  contributions * w_top * w_left * \
                (x_bin >= 0) * (y_bin >= 0)
            add_topright = contributions * w_top * w_right * \
                (x_bin < hog_width - 1) * (y_bin >= 0)
            add_bottomleft = contributions * w_bottom * w_left * \
                (x_bin >= 0) * (y_bin < hog_height - 1)
            add_bottomright = contributions * w_bottom * w_right * \
                (x_bin < hog_width - 1) * (y_bin < hog_height - 1)
            np.add.at(
                hog_flat,
                indices_topleft,
                add_topleft.ravel(),
            )
            np.add.at(
                hog_flat,
                indices_topright,
                add_topright.ravel(),
            )
            np.add.at(
                hog_flat,
                indices_bottomleft,
                add_bottomleft.ravel(),
            )
            np.add.at(
                hog_flat,
                indices_bottomright,
                add_bottomright.ravel(),
            )
            hog = hog_flat.reshape(
                (n_bins,hog_width,hog_height)
            )
            hogs_u.append(hog)

        return np.asarray(hogs_u)


    def compute_discretized_grads(self, images):
        num_orientations = self.num_orientations
        oriented = self.oriented
        n_bins = self.n_bins
        # beware that opencv uses (index, height, width, channel)
        batch_size, n_channel, width, height = images.shape
        # x: width / column, y: height / row

        # compute centered finite difference for pixels that stay one pixel away from the images border
        # compute gradients channel-wise, and then pick the gradient with largest norm
        all_grad_x = images[:,:,2:,1:-1] - images[:,:,:-2,1:-1]
        all_grad_y = images[:,:,1:-1,2:] - images[:,:,1:-1,:-2]
        all_gradnorm2 = all_grad_x ** 2 + all_grad_y ** 2
        max_gradnorm2_channel = np.argmax(all_gradnorm2,axis=1)
        grad_x = np.asarray([
            np.choose(image_max_gradnorm2_channel,image_grad_x)
            for image_max_gradnorm2_channel, image_grad_x in zip(max_gradnorm2_channel,all_grad_x)
        ])
        grad_y = np.asarray([
            np.choose(image_max_gradnorm2_channel,image_grad_y)
            for image_max_gradnorm2_channel, image_grad_y in zip(max_gradnorm2_channel,all_grad_y)
        ])
        grad_norms = np.sqrt(np.max(all_gradnorm2,axis=1))

        half_orientation_weights = np.asarray(
            [
                grad_x * ox + grad_y * oy
                for ox,oy in zip(self.orientation_x, self.orientation_y)
            ]
        ) # shape: (bin_index, batch_index, x, y)

        half_orientations = np.argmax(
            np.abs(half_orientation_weights),
            axis=0,
        )
        if oriented:
            signed_orientation_weights = np.choose(
                half_orientations,
                half_orientation_weights
            )
            orientation_weight_signs = np.sign(signed_orientation_weights)

            # convert to orientations in [0,2*pi]
            orientations = (
                half_orientations + \
                (1 - orientation_weight_signs)/2 * num_orientations
            ).astype(int)
            orientation_weights = np.abs(signed_orientation_weights)
        else:
            orientations = half_orientations
            orientation_weights = half_orientation_weights

        return orientations, orientation_weights, grad_norms

    def generate_edge_map(self,feature,cell_pixels=20):
        """
        Display the HOG feature as an edge map (grayscale)
        """
        N, hog_width, hog_height = feature.shape
        assert N == self.n_bins

        img_cells_all = []
        for y_cell in range(hog_height):
            img_cells_row = []
            for x_cell in range(hog_width):
                histogram = feature[:,x_cell,y_cell]
                img_cell = np.zeros(
                    (cell_pixels,cell_pixels),
                    dtype=float
                )
                for o in range(self.num_orientations):
                    if self.oriented:
                        weight = histogram[o] + histogram[o + self.num_orientations]
                    else:
                        weight = histogram[o]
                    center = (cell_pixels * 0.5, cell_pixels * 0.5)
                    dx = self.orientation_y[o] * 0.5 * cell_pixels
                    dy = - self.orientation_x[o] * 0.5 * cell_pixels
                    p1 = (int(center[0] + dx),int(center[1] + dy))
                    p2 = (int(center[0] - dx),int(center[1] - dy))

                    white = np.asarray([255,255,255])
                    color = tuple(white * weight)
                    img_cell = cv2.line(img_cell,p1,p2,color)
                img_cells_row.append(img_cell)
            img_cells_all.append(
                np.concatenate(img_cells_row,axis=1)
            )
        img_hog = np.concatenate(img_cells_all,axis=0).astype(np.uint8)
        if self.transpose:
            img_hog = np.transpose(img_hog)
        return img_hog

import unittest
