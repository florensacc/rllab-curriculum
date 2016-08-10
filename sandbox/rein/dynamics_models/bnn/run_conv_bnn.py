import numpy as np
import lasagne
from sandbox.rein.dynamics_models.utils import iterate_minibatches, plot_mnist_digit, load_dataset_MNIST, \
    load_dataset_MNIST_plus, load_dataset_Atari

import time
import rllab.misc.logger as logger
import os

os.environ["THEANO_FLAGS"] = "device=gpu"


class Experiment(object):
    def __init__(self, model,
                 ind_softmax,
                 num_epochs,
                 pred_delta,
                 num_bins):
        self.model = model
        self.ind_softmax = ind_softmax
        self.num_epochs = num_epochs
        self.pred_delta = pred_delta
        self.num_bins = num_bins

    def plot_pred_imgs(self, model, inputs, targets, itr, count, ind_softmax, pred_delta):
        # This is specific to Atari.
        import matplotlib.pyplot as plt
        if not hasattr(self, '_fig'):
            self._fig = plt.figure()
            self._fig_1 = self._fig.add_subplot(141)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_2 = self._fig.add_subplot(142)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_3 = self._fig.add_subplot(143)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._fig_4 = self._fig.add_subplot(144)
            plt.tick_params(axis='both', which='both', bottom='off', top='off',
                            labelbottom='off', right='off', left='off', labelleft='off')
            self._im1, self._im2, self._im3, self._im4 = None, None, None, None

        idx = np.random.randint(0, inputs.shape[0], 1)
        pred = []
        for _ in xrange(10):
            pred.append(model.pred_fn(inputs))
        sanity_pred = np.mean(np.array(pred), axis=0)
        input_im = inputs[:, :-3]
        input_im = input_im[idx, :].reshape((1, 42, 42)).transpose(1, 2, 0)[:, :, 0]
        sanity_pred_im = sanity_pred[idx, :-1]
        if ind_softmax:
            sanity_pred_im = sanity_pred_im.reshape((-1, model.num_classes))
            sanity_pred_im = np.argmax(sanity_pred_im, axis=1)
        sanity_pred_im = sanity_pred_im.reshape((1, 42, 42)).transpose(1, 2, 0)[:, :, 0]
        target_im = targets[idx, :-1].reshape((1, 42, 42)).transpose(1, 2, 0)[:, :, 0]

        if pred_delta:
            sanity_pred_im += input_im
            target_im += input_im

        if ind_softmax:
            sanity_pred_im = sanity_pred_im.astype(float) / float(model.num_classes)
            target_im = target_im.astype(float) / float(model.num_classes)
            input_im = input_im.astype(float) / float(model.num_classes)

        err = (1 - np.abs(target_im - sanity_pred_im) * 100.)

        if self._im1 is None or self._im2 is None:
            self._im1 = self._fig_1.imshow(
                input_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=1)
            self._im2 = self._fig_2.imshow(
                target_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=1)
            self._im3 = self._fig_3.imshow(
                sanity_pred_im, interpolation='none', cmap='Greys_r', vmin=0, vmax=1)
            self._im4 = self._fig_4.imshow(
                err, interpolation='none', cmap='Greys_r', vmin=0, vmax=1)
        else:
            self._im1.set_data(input_im)
            self._im2.set_data(target_im)
            self._im3.set_data(sanity_pred_im)
            self._im4.set_data(err)
        act = np.argmax(inputs[idx, -3:][0])
        plt.savefig(
            logger._snapshot_dir + '/dynpred_img_{}_{}_act{}.png'.format(itr, count, act), bbox_inches='tight')

    def train(self, model, num_epochs=500, X_train=None, T_train=None, act=None,
              rew=None, ind_softmax=False, pred_delta=False):

        im_size = X_train.shape[-1]
        X_train = X_train.reshape(-1, im_size * im_size)
        T_train = T_train.reshape(-1, im_size * im_size)
        X = np.hstack((X_train, act))
        Y = np.hstack((T_train, rew))

        logger.log('Training ...')

        for epoch in range(num_epochs):

            # In each epoch, we do a full pass over the training data:
            train_err, train_batches, start_time, kl_values = 0, 0, time.time(), []

            if not model.disable_variance:
                try:
                    print('KL[post||prior]: {}'.format(model.log_p_w_q_w_kl().eval()))
                except Exception:
                    pass

            # Iterate over all minibatches and train on each of them.
            for batch in iterate_minibatches(X, Y, model.batch_size, shuffle=True):
                # Fix old params for KL divergence computation.
                model.save_params()

                # Train current minibatch.
                inputs, targets, _ = batch

                _train_err = model.train_fn(inputs, targets, 1.0)
                # if model.surprise_type == model.SurpriseType.VAR and train_batches % 30 == 0:
                #     print(model.train_update_fn(inputs))
                print(model.fn_l1())

                train_err += _train_err
                train_batches += 1

            pred = []
            for _ in xrange(10):
                pred.append(model.pred_fn(X))
            pred = np.mean(np.array(pred), axis=0)

            pred_im = pred[:, :-1]
            if ind_softmax:
                pred_im = pred_im.reshape((-1, im_size * im_size, model.num_classes))
                pred_im = np.argmax(pred_im, axis=2)

            acc = np.mean(np.sum(np.square(pred_im - Y[:, :-1]), axis=1), axis=0)

            if epoch % 30 == 0:
                self.plot_pred_imgs(model, X, Y, epoch, 1, ind_softmax, pred_delta)

            logger.record_tabular('train loss', train_err / float(train_batches))
            logger.record_tabular('obs err', acc)
            logger.log("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            logger.record_tabular('likelihood std', model.likelihood_sd.eval())

            logger.dump_tabular(with_prefix=False)

        logger.log("Done training.")

    def bin_img(self, lst_img, num_bins):
        for img in lst_img:
            img *= num_bins

    def main(self):

        print("Loading data ...")
        X_train, T_train, act, rew = load_dataset_Atari()
        X_train = X_train[:, np.newaxis, :, :]
        T_train = T_train[:, np.newaxis, :, :]
        if self.ind_softmax:
            self.bin_img(X_train, self.num_bins)
            self.bin_img(T_train, self.num_bins)
            X_train = X_train.astype(int)
            T_train = T_train.astype(int)
            X_train0 = np.tile(X_train[0], reps=[50, 1])
            T_train0 = np.tile(T_train[0], reps=[50, 1])
            X_train1 = np.tile(X_train[1], reps=[50, 1])
            T_train1 = np.tile(T_train[1], reps=[50, 1])
            X_train = np.vstack((X_train0, X_train1))
            T_train = np.vstack((T_train0, T_train1))

        elif self.pred_delta:
            T_train = X_train - T_train

        # Train the model.
        self.train(model=self.model, num_epochs=self.num_epochs, X_train=X_train, T_train=T_train, act=act, rew=rew,
                   ind_softmax=self.ind_softmax, pred_delta=self.pred_delta)
        print('Done.')


if __name__ == '__main__':
    Experiment().main()
