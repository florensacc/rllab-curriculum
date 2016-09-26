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

    def plot_pred_imgs(self, model, counting_table, inputs, targets, itr, count, dir='/imgs'):
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

        # Plotting all images
        for idx in range(inputs.shape[0]):
            sanity_pred = model.pred_fn(inputs)
            cont_emb = model.discrete_emb(inputs)[idx]
            key = np.cast['int'](np.round(cont_emb))
            key_int = self.bin_to_int(key)
            if key_int in counting_table.keys():
                count = counting_table[key_int]
            else:
                count = 0
            # print(key_int, count)
            title_bin = ''.join([str(k) for k in key])
            title_float = ' '.join(['{:.1f}'.format(c) for c in cont_emb])
            title_float2 = ''
            for idy, char in enumerate(title_float):
                title_float2 += char
                if (idy % 101 == 0) and idy != 0:
                    title_float2 += '\n'
            title = title_bin + '\n\n' + title_float2 + '\n\n' + 'count: {}'.format(count)
            plt.suptitle(title)
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

            if not os.path.exists(logger._snapshot_dir + dir):
                os.makedirs(logger._snapshot_dir + dir)
            plt.savefig(
                logger._snapshot_dir + dir + '/model_{}_{}_{}.png'.format(itr, 0, idx), bbox_inches='tight')
