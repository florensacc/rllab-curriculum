import numpy as np


class Plotter:
    def __init__(self):
        pass

    def plot_pred_imgs(self, inputs, targets, itr, count):
        import matplotlib.pyplot as plt
        if not hasattr(self, '_fig'):
            self._fig = plt.figure()
            self._fig_1 = self._fig.add_subplot(241)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_2 = self._fig.add_subplot(242)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_3 = self._fig.add_subplot(243)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_4 = self._fig.add_subplot(244)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_5 = self._fig.add_subplot(245)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_6 = self._fig.add_subplot(246)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_7 = self._fig.add_subplot(247)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_8 = self._fig.add_subplot(248)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._im1, self._im2, self._im3, self._im4 = None, None, None, None
            self._im5, self._im6, self._im7, self._im8 = None, None, None, None

        idx = np.random.randint(0, inputs.shape[0], 1)
        sanity_pred = self.bnn.pred_fn(inputs)
        input_im = inputs[:, :-self.env.spec.action_space.flat_dim]
        lst_input_im = [input_im[idx, i * np.prod(self.bnn.state_dim):(i + 1) * np.prod(self.bnn.state_dim)].reshape(
            self.bnn.state_dim).transpose(1, 2, 0)[:, :, 0] * 256. for i in
                        range(self._num_seq_frames)]
        input_im = input_im[:, -np.prod(self.bnn.state_dim):]
        input_im = input_im[idx, :].reshape(self.bnn.state_dim).transpose(1, 2, 0)[:, :, 0]
        sanity_pred_im = sanity_pred[idx, :-1]
        if self.bnn.output_type == self.bnn.OutputType.CLASSIFICATION:
            sanity_pred_im = sanity_pred_im.reshape((-1, self.bnn.num_classes))
            sanity_pred_im = np.argmax(sanity_pred_im, axis=1)
        sanity_pred_im = sanity_pred_im.reshape(self.bnn.state_dim).transpose(1, 2, 0)[:, :, 0]
        target_im = targets[idx, :-1].reshape(self.bnn.state_dim).transpose(1, 2, 0)[:, :, 0]

        if self._predict_delta:
            sanity_pred_im += input_im
            target_im += input_im

        if self.bnn.output_type == self.bnn.OutputType.CLASSIFICATION:
            sanity_pred_im = sanity_pred_im.astype(float) / float(self.bnn.num_classes)
            target_im = target_im.astype(float) / float(self.bnn.num_classes)
            input_im = input_im.astype(float) / float(self.bnn.num_classes)
            for i in range(len(lst_input_im)):
                lst_input_im[i] = lst_input_im[i].astype(float) / float(self.bnn.num_classes)

        sanity_pred_im *= 256.
        sanity_pred_im = np.around(sanity_pred_im).astype(int)
        target_im *= 256.
        target_im = np.around(target_im).astype(int)
        err = (256 - np.abs(target_im - sanity_pred_im) * 100.)
        input_im *= 256.
        input_im = np.around(input_im).astype(int)

        if self._im1 is None or self._im2 is None:
            self._im1 = self._fig_1.imshow(
                input_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im2 = self._fig_2.imshow(
                target_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im3 = self._fig_3.imshow(
                sanity_pred_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im4 = self._fig_4.imshow(
                err, interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im5 = self._fig_5.imshow(
                lst_input_im[0], interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im6 = self._fig_6.imshow(
                lst_input_im[1], interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im7 = self._fig_7.imshow(
                lst_input_im[2], interpolation='none', cmap='Greys_r', vmin=0, vmax=255)
            self._im8 = self._fig_8.imshow(
                lst_input_im[3], interpolation='none', cmap='Greys_r', vmin=0, vmax=255)

        else:
            self._im1.set_data(input_im)
            self._im2.set_data(target_im)
            self._im3.set_data(sanity_pred_im)
            self._im4.set_data(err)
            self._im5.set_data(lst_input_im[0])
            self._im6.set_data(lst_input_im[1])
            self._im7.set_data(lst_input_im[2])
            self._im8.set_data(lst_input_im[3])
        plt.savefig(
            logger._snapshot_dir + '/dynpred_img_{}_{}.png'.format(itr, count), bbox_inches='tight')
