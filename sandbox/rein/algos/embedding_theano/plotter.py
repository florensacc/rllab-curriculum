import numpy as np
import os
from rllab.misc import logger


class Plotter:
    def __init__(self):
        pass

    def bin_to_int(self, binary):
        integer = 0
        for bit in binary:
            integer = (integer << 1) | bit
        return integer

    def print_embs(self, model, counting_table, inputs, dir='/imgs', hamming_distance=None):
        assert hamming_distance is not None
        arr_cont_emb = model.discrete_emb(inputs)
        with open(logger._snapshot_dir + dir + '/binary_codes.txt', 'w') as bin_codes_file, open(
                                logger._snapshot_dir + dir + '/continuous_codes.txt',
                'w') as cont_codes_file, open(
                            logger._snapshot_dir + dir + '/counts.txt', 'w') as counts_file:
            # Plotting all images
            for idx in range(inputs.shape[0]):
                cont_emb = arr_cont_emb[idx]
                key = np.cast['int'](np.round(cont_emb))
                key_int = self.bin_to_int(key)
                if key_int in counting_table.keys():
                    count = counting_table[key_int]
                    if hamming_distance == 1:
                        for i in range(len(key)):
                            key_trans = np.array(key)
                            key_trans[i] = 1 - key_trans[i]
                            key_trans_int = self.bin_to_int(key_trans)
                            # If you access the counting table directly, it puts a 0, which inflates the size.
                            if key_trans_int in counting_table.keys():
                                count += counting_table[self.bin_to_int(key_trans)]
                else:
                    count = 0

                # --
                # Write-out
                emb_bin_as_str = ''.join([str(k) for k in key])
                emb_cont_as_str = ' '.join(['{:.1f}'.format(c) for c in cont_emb])
                bin_codes_file.write(emb_bin_as_str + '\n')
                cont_codes_file.write(emb_cont_as_str + '\n')
                counts_file.write(str(count) + '\n')

    def print_consistency_embs(self, model, counting_table, inputs, dir='/imgs', hamming_distance=None):
        assert hamming_distance is not None
        arr_cont_emb = model.discrete_emb(inputs)
        # Plotting all images

        for idx in range(inputs.shape[0]):
            with open(logger._snapshot_dir + dir + '/binary_code_{}.txt'.format(idx),
                      'a') as bin_codes_file, open(
                                logger._snapshot_dir + dir + '/continuous_code_{}.txt'.format(idx),
                'a') as cont_codes_file, open(logger._snapshot_dir + dir + '/counts_{}.txt'.format(idx),
                                              'a') as counts_file:
                cont_emb = arr_cont_emb[idx]
                key = np.cast['int'](np.round(cont_emb))
                key_int = self.bin_to_int(key)
                if key_int in counting_table.keys():
                    count = counting_table[key_int]
                    if hamming_distance == 1:
                        for i in range(len(key)):
                            key_trans = np.array(key)
                            key_trans[i] = 1 - key_trans[i]
                            key_trans_int = self.bin_to_int(key_trans)
                            # If you access the counting table directly, it puts a 0, which inflates the size.
                            if key_trans_int in counting_table.keys():
                                count += counting_table[self.bin_to_int(key_trans)]
                else:
                    count = 0

                # --
                # Write-out
                emb_bin_as_str = ''.join([str(k) for k in key])
                emb_cont_as_str = ' '.join(['{:.1f}'.format(c) for c in cont_emb])
                bin_codes_file.write(emb_bin_as_str + '\n')
                cont_codes_file.write(emb_cont_as_str + '\n')
                counts_file.write(str(count) + '\n')

    def plot_pred_imgs(self, model, inputs, targets, itr, dir='/imgs'):
        import matplotlib.pyplot as plt
        if not hasattr(self, '_fig'):
            self._fig = plt.figure()
            self._fig_2 = self._fig.add_subplot(131)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_3 = self._fig.add_subplot(132)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_4 = self._fig.add_subplot(133)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._im1, self._im2, self._im3, self._im4 = None, None, None, None

        if not os.path.exists(logger._snapshot_dir + dir):
            os.makedirs(logger._snapshot_dir + dir)

        # Predict all images at once.
        sanity_pred = model.pred_fn(inputs)
        # Plotting all images
        for idx in range(inputs.shape[0]):
            sanity_pred_im = sanity_pred[idx, :]
            if model.output_type == model.OutputType.CLASSIFICATION:
                sanity_pred_im = sanity_pred_im.reshape((-1, model.num_classes))
                sanity_pred_im = np.argmax(sanity_pred_im, axis=1)
            sanity_pred_im = sanity_pred_im.reshape(model.state_dim).transpose(1, 2, 0)[:, :, 0]
            target_im = targets[idx, :].reshape(model.state_dim).transpose(1, 2, 0)[:, :, 0]

            if model.output_type == model.OutputType.CLASSIFICATION:
                sanity_pred_im = sanity_pred_im.astype(float) / float(model.num_classes)
                target_im = target_im.astype(float) / float(model.num_classes)

            sanity_pred_im *= 256.
            sanity_pred_im = np.around(sanity_pred_im).astype(int)
            target_im *= 256.
            target_im = np.around(target_im).astype(int)
            err = (256 - np.abs(target_im - sanity_pred_im) * 100.)

            if self._im1 is None or self._im2 is None:
                self._im2 = self._fig_2.imshow(
                    target_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
                self._im3 = self._fig_3.imshow(
                    sanity_pred_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
                self._im4 = self._fig_4.imshow(
                    err, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)

            else:
                self._im2.set_data(target_im)
                self._im3.set_data(sanity_pred_im)
                self._im4.set_data(err)

            plt.savefig(
                logger._snapshot_dir + dir + '/model_{}_{}.png'.format(itr, idx), bbox_inches='tight')
