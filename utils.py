import tensorflow as tf


def get_weight(shape):
    """
    Create a trainable weight variable with the given shape
    Args:
        shape: The size of the weight matrix to be created
    Returns:
        weight: weight variable
    """
    init = tf.truncated_normal_initializer(stddev=0.02)
    weight = tf.get_variable('weight', shape, initializer=init, trainable=True)
    return weight


def get_bias(shape):
    """
    Create a bias variable with the given shape
    Args:
        shape: shape of the variable to be created
    Returns:
        bias: bias variable
    """
    init = tf.zeros_initializer()
    bias = tf.get_variable('bias', shape, initializer=init, trainable=True)
    return bias


def fc_layer(x, h_units, var_scope='fc_layer'):
    """
    Create a fully connected layer
    Args:
        x: input array of size [Batch, Features]
        h_units: number of neurons in the hidden layer (integer)
        var_scope: scope of the layer variables (string)
    Returns:
        out: output array of size [Batch, h_units]
    """
    x_shape = x.get_shape().as_list()  # get input shape as list
    with tf.variable_scope(var_scope):
        weight = get_weight([x_shape[-1], h_units])
        bias = get_bias([h_units])
        out = tf.matmul(x, weight)
        out = tf.nn.bias_add(out, bias)
    return out


def conv2d(x, num_kernels, kernel_size, stride=1, var_scope='conv', add_bias=True):
    """
    Create a 2D convolution layer
    Args:
        x: input array of size [Batch, Height, Width, Channels]
        num_kernels: number of kernels (integer)
        kernel_size: size of kernel (integer--->assumming square kernel size)
        stride: step size of the kernel (integer)
        var_scope: scope of layer variables (string)
        add_bias: add and use bias variable (boolean)
    Returns:
        out: output array of a 2D convolution operation of size
            [Batch, H/stride, W/stride, num_kernels]
    """
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(var_scope):
        weight_shape = [kernel_size, kernel_size, x_shape[-1], num_kernels]
        weight = get_weight(weight_shape)
        conv = tf.nn.conv2d(x, filter=weight, strides=stride, padding='SAME')

        if add_bias:
            bias_shape = [num_kernels]
            bias = get_bias(bias_shape)
            conv = tf.nn.bias_add(conv, bias)
    return conv


def batch_norm(x, state, var_scope='bn'):
    """
    Create a batch normalization (global) layer
    Args:
        x: input ND array (N>=2)
        state: whether the network is in training or test mode (boolean)
        var_scope: scope of the variables (string)

    Returns:
        out: x_norm: a normalized value of the input
    """
    with tf.variable_scope(var_scope):
        # create beta and gamma variables
        x_shape = x.get_shape().as_list()
        var_shape = [1]*(len(x_shape)-1) + [x[-1]]
        beta = tf.get_variable('beta', shape=var_shape,
                               initializer=tf.zeros_initializer, trainable=True)
        gamma = tf.get_variable('gamma', shape=var_shape,
                                initializer=tf.ones_initializer, trainable=True)
        # calculate mini-batch mean and variance
        batch_mu, batch_var = tf.nn.moments(x, axis=(0, 1, 2), keep_dims=True)
        # create exponential moving average to update mean and variance stat
        moving_avrg = tf.train.ExponentialMovingAverage(decay=0.99)

        def update_stat():
            """
            Create a copy of the batch_mu and batch_var variables and update
            them
            """
            apply_moving_avrg = moving_avrg.apply([batch_mu, batch_var])
            with tf.control_dependencies([apply_moving_avrg]):
                return tf.identity(batch_mu), tf.identity(batch_var)
        mean, var = tf.cond(state, update_stat,
                            lambda: (moving_avrg.average(batch_mu),
                                     moving_avrg.average(batch_var)))
        x_norm = tf.nn.batch_normalization(x, mean, var, beta, gamma,
                                           variance_epsilon=1e-5)
    return x_norm


def resnet_unit(x, num_kernels, bn_state, stride=1, var_scope='resnet'):
    """
    Create a resnet block

    Args:
        x: input feature map of size [Batch, Height, Width, In_Channel]
        num_kernels: number of filters (integer)
        bn_state: batch normalization state (boolean)
        stride: kernel step (integer, only applied to the first convolution)
        var_scope: scope of the variables created

    Returns:
        out: feature map with size [Batch, Height, Width, Out_Channel]
    """
    with tf.variable_scope(var_scope):
        conv_one = conv2d(x, num_kernels, 3, stride=stride,
                          var_scope='conv_one')
        bn_one = batch_norm(conv_one, bn_state, var_scope='bn_one')
        lrelu_one = tf.nn.leaky_relu(bn_one)
        conv_two = conv2d(lrelu_one, num_kernels, 3, var_scope='conv_two')
        bn_two = batch_norm(conv_two, bn_state, var_scope='bn_two')
        lrelu_two = tf.nn.leaky_relu(bn_two)
        # add shortcut
        if stride > 1:
            x_short_cut = conv2d(
                x, num_kernels, 1, stride=stride, var_scope='short_cut')
            out = tf.add(x_short_cut, lrelu_two)
        else:
            out = tf.add(x, lrelu_two)
        return out
