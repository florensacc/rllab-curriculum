import tensorflow as tf
import numpy as np
from sandbox.bradly.third_person.discriminators.flip_gradients import flip_gradient


class Discriminator(object):
    def __init__(self, input_dim, output_dim_class=2, output_dim_dom=None):
        self.input_dim = input_dim
        self.output_dim_class = output_dim_class
        self.output_dim_dom = output_dim_dom
        self.learning_rate = 0.001
        self.loss = None
        self.discrimination_logits = None
        self.optimizer = None
        self.nn_input = None
        self.class_target = None
        self.sess = None

    def init_tf(self):
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def make_network(self, dim_input, output_dim_class, output_dim_dom):
        raise NotImplementedError

    def train(self, data_batch, targets_batch):
        cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.nn_input: data_batch,
                                                                     self.class_target: targets_batch})[1]
        return cost

    def __call__(self, data, softmax=True):
        if softmax is True:
            logits = tf.nn.softmax(self.discrimination_logits)
        else:
            logits = self.discrimination_logits
        log_prob = self.sess.run([logits], feed_dict={self.nn_input: data})[0]
        return log_prob

    @staticmethod
    def init_weights(shape, name=None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

    @staticmethod
    def init_bias(shape, name=None):
        return tf.Variable(tf.zeros(shape, dtype='float'), name=name)

    @staticmethod
    def conv2d(img, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))

    @staticmethod
    def max_pool(img, k):
        return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def get_mlp_layers(self, mlp_input, number_layers, dimension_hidden, name_prefix='', dropout=None):
        """compute MLP with specified number of layers.
            math: sigma(Wx + b)
            for each layer, where sigma is by default relu"""
        cur_top = mlp_input
        for layer_step in range(0, number_layers):
            in_shape = cur_top.get_shape().dims[1].value
            cur_weight = self.init_weights([in_shape, dimension_hidden[layer_step]],
                                           name='w_' + name_prefix + str(layer_step))
            cur_bias = self.init_bias([dimension_hidden[layer_step]],
                                      name='b_' + name_prefix + str(layer_step))
            if layer_step != number_layers-1:  # final layer has no RELU
                cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
                if dropout is not None:
                    cur_top = tf.nn.dropout(cur_top, dropout)
            else:
                cur_top = tf.matmul(cur_top, cur_weight) + cur_bias
        return cur_top

    @staticmethod
    def get_xavier_weights(filter_shape, poolsize=(2, 2)):
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))

        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(filter_shape, minval=low, maxval=high, dtype=tf.float32))
        #return tf.Variable(tf.random_normal(filter_shape, mean=0.01, stddev=0.001, dtype=tf.float32))

    def get_loss_layer(self, pred, target_output):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, target_output))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return cost, optimizer


class MLPDiscriminator(Discriminator):
    def __init__(self, input_dim):
        super(MLPDiscriminator, self).__init__(input_dim)
        self.make_network(dim_input=input_dim, dim_output=2)
        self.init_tf()

    def make_network(self, dim_input, dim_output):
        n_layers = 3
        dim_hidden = (n_layers - 1) * [40]
        dim_hidden.append(dim_output)
        nn_input, target = self.get_input_layer(dim_input, dim_output)
        mlp_applied = self.get_mlp_layers(nn_input, n_layers, dim_hidden)
        loss, optimizer = self.get_loss_layer(pred=mlp_applied, target_output=target)
        self.class_target = target
        self.nn_input = nn_input
        self.discrimination_logits = mlp_applied
        self.optimizer = optimizer
        self.loss = loss

    def get_input_layer(self, dim_input, dim_output):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        net_input = tf.placeholder('float', [None, dim_input], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets


class ConvDiscriminator(Discriminator):

    def __init__(self, input_dim):
        super(ConvDiscriminator, self).__init__(input_dim)
        self.make_network(dim_input=input_dim, dim_output=2)
        self.init_tf()

    def make_network(self, dim_input, dim_output):
        """
        An example a network in tf that has both state and image inputs.

        Args:
            dim_input: Dimensionality of input.
            dim_output: Dimensionality of the output.
            batch_size: Batch size.
            network_config: dictionary of network structure parameters
        Returns:
            A tfMap object that stores inputs, outputs, and scalar loss.
        """
        n_mlp_layers = 2
        layer_size = 128
        dim_hidden = (n_mlp_layers - 1) * [layer_size]
        dim_hidden.append(dim_output)
        pool_size = 2
        filter_size = 3
        im_width = dim_input[0]
        im_height = dim_input[1]
        num_channels = dim_input[2]
        num_filters = [5, 5]

        nn_input, target = self.get_input_layer(im_width, im_height, num_channels, dim_output)

        # we pool twice, each time reducing the image size by a factor of 2.
        #conv_out_size = int(im_width * im_height * num_filters[1] / ((2.0 * pool_size) * (2.0 * pool_size)))
        conv_out_size = int(im_width * im_height * num_filters[1] / (2.0 * pool_size))
        first_dense_size = conv_out_size

        # Store layers weight & bias
        weights = {
            'wc1': self.get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size)),
        # 5x5 conv, 1 input, 32 outputs
            'wc2': self.get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size)),
        # 5x5 conv, 32 inputs, 64 outputs
        }

        biases = {
            'bc1': self.init_bias([num_filters[0]]),
            'bc2': self.init_bias([num_filters[1]]),
        }

        conv_layer_0 = self.conv2d(img=nn_input, w=weights['wc1'], b=biases['bc1'])

        conv_layer_0 = self.max_pool(conv_layer_0, k=pool_size)

        #conv_layer_1 = self.conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])

        #conv_layer_1 = self.max_pool(conv_layer_1, k=pool_size)

        conv_layer_1 = conv_layer_0

        conv_out_flat = tf.reshape(conv_layer_1, [-1, conv_out_size])

        #fc_input = tf.concat(concat_dim=1, values=[conv_out_flat, state_input])

        fc_output = self.get_mlp_layers(conv_out_flat, n_mlp_layers, dim_hidden, dropout=None)

        loss, optimizer = self.get_loss_layer(pred=fc_output, target_output=target)

        #trainable_vars = tf.trainable_variables()
        #loss_with_reg = loss
        #for var in trainable_vars:
        #    loss_with_reg += 0.0005*tf.nn.l2_loss(var)

        self.class_target = target
        self.nn_input = nn_input
        self.discrimination_logits = fc_output
        self.optimizer = optimizer
        self.loss = loss

    @staticmethod
    def get_input_layer(im_width, im_height, num_channels, dim_output=2):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn.
        precision: precision matrix used to commpute loss."""
        net_input = tf.placeholder('float', [None, im_width, im_height, num_channels], name='nn_input')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input, targets


class VelocityDiscriminator(Discriminator):
    def __init__(self, input_dim):
        super(VelocityDiscriminator, self).__init__(input_dim=input_dim)
        self.make_network(dim_input=input_dim, dim_output=2)
        self.init_tf()

    def make_network(self, dim_input, dim_output):
        """
        An example a network in tf that has both state and image inputs.

        Args:
            dim_input: Dimensionality of input.
            dim_output: Dimensionality of the output.
            batch_size: Batch size.
            network_config: dictionary of network structure parameters
        Returns:
            A tfMap object that stores inputs, outputs, and scalar loss.
        """
        n_mlp_layers = 4
        layer_size = 128
        dim_hidden = (n_mlp_layers - 1) * [layer_size]
        dim_hidden.append(dim_output)
        pool_size = 2
        filter_size = 3
        im_width = dim_input[0]
        im_height = dim_input[1]
        num_channels = dim_input[2]
        num_filters = [5, 5]

        nn_input_one, nn_input_two, target = self.get_input_layer(im_width, im_height, num_channels, dim_output)

        # we pool twice, each time reducing the image size by a factor of 2.
        #conv_out_size = int(im_width / (2.0 * pool_size) * im_height / (2.0 * pool_size) * num_filters[1])
        conv_out_size = int(im_width * im_height * num_filters[1] / (2.0 * pool_size))
        first_dense_size = conv_out_size

        # Store layers weight & bias
        weights = {
            'wc1': self.get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size)),
        # 5x5 conv, 1 input, 32 outputs
            #'wc2': self.get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size)),
        # 5x5 conv, 32 inputs, 64 outputs
        }

        biases = {
            'bc1': self.init_bias([num_filters[0]]),
            'bc2': self.init_bias([num_filters[1]]),
        }

        conv_layer_0_input_one = self.conv2d(img=nn_input_one, w=weights['wc1'], b=biases['bc1'])

        conv_layer_0_input_two = self.conv2d(img=nn_input_two, w=weights['wc1'], b=biases['bc1'])

        conv_layer_0_input_one = self.max_pool(conv_layer_0_input_one, k=pool_size)

        conv_layer_0_input_two = self.max_pool(conv_layer_0_input_two, k=pool_size)

        #conv_layer_1_input_one = self.conv2d(img=conv_layer_0_input_one, w=weights['wc2'], b=biases['bc2'])

        #conv_layer_1_input_two = self.conv2d(img=conv_layer_0_input_two, w=weights['wc2'], b=biases['bc2'])

        #conv_layer_1_input_one = self.max_pool(conv_layer_1_input_one, k=pool_size)

        #conv_layer_1_input_two = self.max_pool(conv_layer_1_input_two, k=pool_size)

        conv_out_flat_input_one = tf.reshape(conv_layer_0_input_one, [-1, conv_out_size])

        conv_out_flat_input_two = tf.reshape(conv_layer_0_input_two, [-1, conv_out_size])

        cur_top = conv_out_flat_input_one
        in_shape = conv_out_size
        cur_weight = self.init_weights([in_shape, layer_size], name='w_feats_one')
        cur_bias = self.init_bias([layer_size], name='b_feats_one')
        conv_one_features = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)

        cur_top = conv_out_flat_input_two
        #in_shape = conv_out_size
        #cur_weight = self.init_weights([in_shape, layer_size], name='w_feats_two')
        #cur_bias = self.init_bias([layer_size], name='b_feats_two')
        conv_two_features = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)

        fc_input = tf.concat(concat_dim=1, values=[conv_one_features, conv_two_features])
        #fc_input = conv_one_features - conv_two_features #try concat and RNN here.

        #fc_input = tf.concat(concat_dim=1, values=[conv_out_flat, state_input])

        fc_output = self.get_mlp_layers(fc_input, n_mlp_layers, dim_hidden)

        loss, optimizer = self.get_loss_layer(pred=fc_output, target_output=target)

        self.class_target = target
        self.nn_input = [nn_input_one, nn_input_two]
        self.discrimination_logits = fc_output
        self.optimizer = optimizer
        self.loss = loss

    @staticmethod
    def get_input_layer(im_width, im_height, num_channels, dim_output=2):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn."""
        net_input_one = tf.placeholder('float', [None, im_width, im_height, num_channels], name='nn_input_one')
        net_input_two = tf.placeholder('float', [None, im_width, im_height, num_channels], name='nn_input_two')
        targets = tf.placeholder('float', [None, dim_output], name='targets')
        return net_input_one, net_input_two, targets

    def train(self, data_batch, targets_batch):
        if len(data_batch) != 2:
            raise ValueError('data batch should have length two')
        cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.nn_input[0]: data_batch[0],
                                                                     self.nn_input[1]: data_batch[1],
                                                                     self.class_target: targets_batch})[1]
        return cost

    def __call__(self, data, softmax=True):
        if len(data) != 2:
            raise ValueError('data size is wrong')
        if softmax is True:
            log_prob = self.sess.run([tf.nn.softmax(self.discrimination_logits)],
                                     feed_dict={self.nn_input[0]: data[0], self.nn_input[1]: data[1]})[0]
        else:
            log_prob = self.sess.run([self.discrimination_logits],
                                     feed_dict={self.nn_input[0]: data[0], self.nn_input[1]: data[1]})[0]
        return log_prob


class DomainConfusionVelocityDiscriminator(Discriminator):
    def __init__(self, input_dim):
        super(DomainConfusionVelocityDiscriminator, self).__init__(input_dim)
        self.dom_targets = None
        self.dom_logits = None
        self.make_network(input_dim, [2, 2])
        self.init_tf()

    def make_network(self, dim_input, dim_output):
        """
        One loss given by the class error, expert demo vs policy
        One loss given by domain class error, which domain were the samples collected from
        The domain class error is trained with gradient ascent, that is we destroy information useful for
        classifying the domain from the conv layers. This helps to learn domain neutral classification.
        """
        n_mlp_layers = 4
        layer_size = 128
        dim_hidden_class = (n_mlp_layers - 1) * [layer_size]
        dim_hidden_class.append(dim_output[0])
        dim_hidden_dom = (n_mlp_layers - 1) * [layer_size]
        dim_hidden_dom.append(dim_output[1])
        pool_size = 2
        filter_size = 3
        im_width = dim_input[0]
        im_height = dim_input[1]
        num_channels = dim_input[2]
        num_filters = [5, 5]

        # We make a velocity discriminator as before, but then use a domain discriminator with shared weights
        # Gradient ascent is done on the domain calculations, making the network domain-blind.

        nn_input_one, nn_input_two, targets, domain_targets = self.get_input_layer(im_width, im_height,
                                                                                   num_channels,
                                                                                   dim_output_class=dim_output[0],
                                                                                   dim_output_dom=dim_output[1]
                                                                                   )

        # we pool twice, each time reducing the image size by a factor of 2.
        #conv_out_size = int(im_width / (2.0 * pool_size) * im_height / (2.0 * pool_size) * num_filters[1])
        conv_out_size = int(im_width * im_height * num_filters[1] / (2.0 * pool_size))
        first_dense_size = conv_out_size

        # Store layers weight & bias
        weights = {
            'wc1': self.get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size)),
        # 5x5 conv, 1 input, 32 outputs
            'wc2': self.get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size)),
        # 5x5 conv, 32 inputs, 64 outputs
        }

        biases = {
            'bc1': self.init_bias([num_filters[0]]),
            'bc2': self.init_bias([num_filters[1]]),
        }

        conv_layer_0_input_one = self.conv2d(img=nn_input_one, w=weights['wc1'], b=biases['bc1'])

        conv_layer_0_input_two = self.conv2d(img=nn_input_two, w=weights['wc1'], b=biases['bc1'])

        conv_layer_0_input_one = self.max_pool(conv_layer_0_input_one, k=pool_size)

        conv_layer_0_input_two = self.max_pool(conv_layer_0_input_two, k=pool_size)

        #conv_layer_1_input_one = self.conv2d(img=conv_layer_0_input_one, w=weights['wc2'], b=biases['bc2'])

        #conv_layer_1_input_two = self.conv2d(img=conv_layer_0_input_two, w=weights['wc2'], b=biases['bc2'])

        #conv_layer_1_input_one = self.max_pool(conv_layer_1_input_one, k=pool_size)

        #conv_layer_1_input_two = self.max_pool(conv_layer_1_input_two, k=pool_size)

        conv_out_flat_input_one = tf.reshape(conv_layer_0_input_one, [-1, conv_out_size])

        conv_out_flat_input_two = tf.reshape(conv_layer_0_input_two, [-1, conv_out_size])

        cur_top = conv_out_flat_input_one
        in_shape = conv_out_size
        feat_weight = self.init_weights([in_shape, layer_size], name='w_feats_one')
        feat_bias = self.init_bias([layer_size], name='b_feats_one')
        conv_one_features = tf.nn.relu(tf.matmul(cur_top, feat_weight) + feat_bias)

        cur_top = conv_out_flat_input_two
        #in_shape = conv_out_size
        #ur_weight = self.init_weights([in_shape, layer_size], name='w_feats_two')
        #cur_bias = self.init_bias([layer_size], name='b_feats_two')
        conv_two_features = tf.nn.relu(tf.matmul(cur_top, feat_weight) + feat_bias)


        #fc_input = conv_one_features - conv_two_features #try concat and RNN here.

        #fc_input = tf.concat(concat_dim=1, values=[conv_out_flat, state_input])

        fc_input = tf.concat(concat_dim=1, values=[conv_one_features, conv_two_features])

        fc_output = self.get_mlp_layers(fc_input, n_mlp_layers, dim_hidden_class, name_prefix='targets')

        class_loss = self.get_loss_layer(pred=fc_output, target_output=targets)

        self.class_target = targets
        self.nn_input = [nn_input_one, nn_input_two]
        self.discrimination_logits = fc_output
        #self.optimizer = optimizer
        #self.loss = loss

        # Domain confusion

        conv_layer_0 = self.conv2d(img=nn_input_one, w=weights['wc1'], b=biases['bc1'])

        conv_layer_0 = self.max_pool(conv_layer_0, k=pool_size)

        #conv_layer_1 = self.conv2d(img=conv_layer_0, w=weights['wc2'], b=biases['bc2'])

        #conv_layer_1 = self.max_pool(conv_layer_1, k=pool_size)

        conv_domain_flat = tf.reshape(conv_layer_0, [-1, conv_out_size])

        domain_features = tf.nn.relu(tf.matmul(conv_domain_flat, feat_weight) + feat_bias)

        domain_features_flipped = flip_gradient(domain_features)

        domain_mlp_out = self.get_mlp_layers(domain_features_flipped, n_mlp_layers, dim_hidden_dom, name_prefix='dom')
        dom_loss = self.get_loss_layer(domain_mlp_out, domain_targets)
        self.dom_targets = domain_targets
        self.dom_logits = domain_mlp_out

        self.loss = dom_loss + class_loss
        self.optimizer = self.get_optimizer(self.loss)

    def get_loss_layer(self, pred, target_output):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, target_output))
        return cost

    def get_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return optimizer

    @staticmethod
    def get_input_layer(im_width, im_height, num_channels, dim_output_dom, dim_output_class=2):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn."""
        net_input_one = tf.placeholder('float', [None, im_width, im_height, num_channels], name='nn_input_one')
        net_input_two = tf.placeholder('float', [None, im_width, im_height, num_channels], name='nn_input_two')
        class_targets = tf.placeholder('float', [None, dim_output_class], name='class_targets')
        domain_targets = tf.placeholder('float', [None, dim_output_dom], name='domain_targets')
        return net_input_one, net_input_two, class_targets, domain_targets

    def train(self, data_batch, targets_batch):
        class_labels = targets_batch[0]
        domain_labels = targets_batch[1]
        nn_input_image_one = data_batch[0]
        nn_input_image_two = data_batch[1]
        return self.sess.run([self.optimizer, self.loss], feed_dict={self.nn_input[0]: nn_input_image_one,
                                                                     self.nn_input[1]: nn_input_image_two,
                                                                     self.dom_targets: domain_labels,
                                                                     self.class_target: class_labels})[1]

    def __call__(self, data, softmax=True):
        if len(data) != 2:
            raise ValueError('data size is wrong')
        if softmax is True:
            log_prob = self.sess.run([tf.nn.softmax(self.discrimination_logits)],
                                     feed_dict={self.nn_input[0]: data[0], self.nn_input[1]: data[1]})[0]
        else:
            log_prob = self.sess.run([self.discrimination_logits],
                                     feed_dict={self.nn_input[0]: data[0], self.nn_input[1]: data[1]})[0]
        return log_prob

    def test_domain_disctinction(self, data, softmax=True):
        if softmax is True:
            log_prob = self.sess.run([tf.nn.softmax(self.discrimination_logits)],
                                     feed_dict={self.nn_input[0]: data})[0]
        else:
            log_prob = self.sess.run([self.discrimination_logits],
                                     feed_dict={self.nn_input[0]: data[0], self.nn_input[1]: data[1]})[0]
        return log_prob


class DomainConfusionDiscriminator(Discriminator):
    def __init__(self, input_dim, output_dim_class, output_dim_dom):
        super(DomainConfusionDiscriminator, self).__init__(input_dim)
        self.dom_targets = None
        self.dom_logits = None
        self.label_accuracy = None
        self.dom_accuracy = None
        if output_dim_class is None:
            output_dim_class = 2
            output_dim_dom = 2
        else:
            output_dim_class = output_dim_class
            output_dim_dom = output_dim_dom
        self.make_network(input_dim, output_dim_class=output_dim_class, output_dim_dom=output_dim_dom)
        self.init_tf()

    def make_network(self, dim_input, output_dim_class, output_dim_dom):
        """
        One loss given by the class error, expert demo vs policy
        One loss given by domain class error, which domain were the samples collected from
        The domain class error is trained with gradient ascent, that is we destroy information useful for
        classifying the domain from the conv layers. This helps to learn domain neutral classification.
        """
        n_mlp_layers = 3
        layer_size = 128
        dim_hidden_class = (n_mlp_layers - 1) * [layer_size]
        dim_hidden_class.append(output_dim_class)
        dim_hidden_dom = (n_mlp_layers - 1) * [layer_size]
        dim_hidden_dom.append(output_dim_dom)
        pool_size = 2
        filter_size = 3
        im_width = dim_input[0]
        im_height = dim_input[1]
        num_channels = dim_input[2]
        num_filters = [32, 48]

        # We make a velocity discriminator as before, but then use a domain discriminator with shared weights
        # Gradient ascent is done on the domain calculations, making the network domain-blind.

        nn_input_one, targets, domain_targets = self.get_input_layer(im_width, im_height, num_channels,
                                                                     dim_output_class=output_dim_class,
                                                                     dim_output_dom=output_dim_dom)

        # we pool twice, each time reducing the image size by a factor of 2.
        #conv_out_size = int(im_width * im_height / (pool_size * pool_size) * num_filters[1])
        #conv_out_size = int(im_width * im_height * num_filters[1] / (2.0 * pool_size))
        #first_dense_size = conv_out_size

        # Store layers weight & bias
        weights = {
            'wc1': self.get_xavier_weights([filter_size, filter_size, num_channels, num_filters[0]], (pool_size, pool_size)),
        # 5x5 conv, 1 input, 32 outputs
            'wc2': self.get_xavier_weights([filter_size, filter_size, num_filters[0], num_filters[1]], (pool_size, pool_size)),
        # 5x5 conv, 32 inputs, 48 outputs
        }

        biases = {
            'bc1': self.init_bias([num_filters[0]]),
            'bc2': self.init_bias([num_filters[1]]),
        }

        conv_layer_0_input_one = self.conv2d(img=nn_input_one, w=weights['wc1'], b=biases['bc1'])

        conv_layer_0_input_one = self.max_pool(conv_layer_0_input_one, k=pool_size)

        conv_layer_1_input_one = self.conv2d(img=conv_layer_0_input_one, w=weights['wc2'], b=biases['bc2'])

        conv_layer_1_input_one = self.max_pool(conv_layer_1_input_one, k=pool_size)

        shp = conv_layer_1_input_one.get_shape().as_list()
        conv_out_size = shp[1]*shp[2]*shp[3]

        conv_out_flat_input_one = tf.reshape(conv_layer_1_input_one, [-1, conv_out_size])

        cur_top = conv_out_flat_input_one
        in_shape = conv_out_size
        feat_weight = self.init_weights([in_shape, layer_size], name='w_feats_one')
        feat_bias = self.init_bias([layer_size], name='b_feats_one')
        conv_one_features = tf.nn.relu(tf.matmul(cur_top, feat_weight) + feat_bias)

        self.conv_one_feats = conv_one_features

        fc_output = self.get_mlp_layers(conv_one_features, n_mlp_layers, dim_hidden_class, name_prefix='targets')

        class_loss = self.get_loss_layer(pred=fc_output, target_output=targets)

        self.class_target = targets
        self.nn_input = nn_input_one
        self.discrimination_logits = fc_output

        label_accuracy = tf.equal(tf.argmax(self.class_target, 1),
                                       tf.argmax(tf.nn.softmax(self.discrimination_logits), 1))
        self.label_accuracy = tf.reduce_mean(tf.cast(label_accuracy, tf.float32))

        # Domain confusion

        domain_features_flipped = flip_gradient(conv_one_features)

        domain_mlp_out = self.get_mlp_layers(domain_features_flipped, n_mlp_layers, dim_hidden_dom, name_prefix='dom')
        dom_loss = self.get_loss_layer(domain_mlp_out, domain_targets)
        self.dom_targets = domain_targets
        self.dom_logits = domain_mlp_out

        dom_accuracy = tf.equal(tf.argmax(self.dom_targets, 1),
                                tf.argmax(tf.nn.softmax(self.dom_logits), 1))
        self.dom_accuracy = tf.reduce_mean(tf.cast(dom_accuracy, tf.float32))

        # Final loss and optimizer
        self.loss = dom_loss + class_loss
        self.optimizer = self.get_optimizer(self.loss)

    def get_loss_layer(self, pred, target_output):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, target_output))
        return cost

    def get_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return optimizer

    @staticmethod
    def get_input_layer(im_width, im_height, num_channels, dim_output_dom, dim_output_class):
        """produce the placeholder inputs that are used to run ops forward and backwards.
        net_input: usually an observation.
        action: mu, the ground truth actions we're trying to learn."""
        net_input_one = tf.placeholder('float', [None, im_width, im_height, num_channels], name='nn_input_one')
        class_targets = tf.placeholder('float', [None, dim_output_class], name='class_targets')
        domain_targets = tf.placeholder('float', [None, dim_output_dom], name='domain_targets')
        return net_input_one, class_targets, domain_targets

    def train(self, data_batch, targets_batch):
        class_labels = targets_batch[0]
        domain_labels = targets_batch[1]
        nn_input_image_one = data_batch
        return self.sess.run([self.optimizer, self.loss], feed_dict={self.nn_input: nn_input_image_one,
                                                                     self.dom_targets: domain_labels,
                                                                     self.class_target: class_labels})[1]

    def get_dom_accuracy(self, data, dom_labels):
        return self.sess.run([self.dom_accuracy], feed_dict={self.nn_input: data,
                                                             self.dom_targets: dom_labels})[0]

    def get_lab_accuracy(self, data, class_labels):
        return self.sess.run([self.label_accuracy], feed_dict={self.nn_input: data,
                                                               self.class_target: class_labels})[0]

    def __call__(self, data, softmax=True):
        if softmax is True:
            log_prob = self.sess.run([tf.nn.softmax(self.discrimination_logits)],
                                     feed_dict={self.nn_input: data})[0]
        else:
            log_prob = self.sess.run([self.discrimination_logits],
                                     feed_dict={self.nn_input: data})[0]
        return log_prob

    def get_reward(self, data, softmax=True):
        if softmax is True:
            log_prob = self.sess.run([tf.nn.softmax(self.discrimination_logits)],
                                     feed_dict={self.nn_input: data})[0]
        else:
            log_prob = self.sess.run([self.discrimination_logits],
                                     feed_dict={self.nn_input: data})[0]
        return log_prob

    def get_conv_one_feats(self, data):
        # obtain the domain invariant features.
        return self.sess.run([self.conv_one_feats], feed_dict={self.nn_input: data})[0]

    encode = get_conv_one_feats



