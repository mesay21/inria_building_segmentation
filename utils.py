import glob
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


def conv2d(x, num_kernels, kernel_size, stride=1, var_scope='conv', add_bias=False):
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
        var_shape = [1]*(len(x_shape)-1) + [x_shape[-1]]
        beta = tf.get_variable('beta', shape=var_shape,
                               initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.get_variable('gamma', shape=var_shape,
                                initializer=tf.ones_initializer(), trainable=True)
        # calculate mini-batch mean and variance
        batch_mu, batch_var = tf.nn.moments(x, axes=(0, 1, 2), keep_dims=True)
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
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(var_scope):
        conv_one = conv2d(x, num_kernels, 3, stride=stride,
                          var_scope='conv_one')
        bn_one = batch_norm(conv_one, bn_state, var_scope='bn_one')
        lrelu_one = lrelu(bn_one)
        conv_two = conv2d(lrelu_one, num_kernels, 3, var_scope='conv_two')
        bn_two = batch_norm(conv_two, bn_state, var_scope='bn_two')
        # add shortcut
        if (stride > 1 or x_shape[-1] != num_kernels):
            x_short_cut = conv2d(
                x, num_kernels, 1, stride=stride, var_scope='short_cut')
            bn_short_cut = batch_norm(
                x_short_cut, bn_state, var_scope='bn_short_cut')
            out = tf.add(bn_short_cut, bn_two)
        else:
            out = tf.add(x, bn_two)
        return lrelu(out)


def lrelu(x):
    """
    Create a leaky relu layer
    Args:
        x: input array
    Returns:
        apply leaky relu activation to the input
    """
    return tf.nn.leaky_relu(x)


def upsample(x, size=2):
    """
    Create upsampling layer
    Args:
        x: input array of size [Batch, Height, Width, Channels]
    Returns:
        out: bicubic upsampled fature map of size
            [Batch, size*Height, size*Width, Channels]
    """
    im_size = tf.shape(x)[1:3] * \
        tf.constant(size)  # new height and width values
    out = tf.image.resize(x, im_size)
    return out


def iterator(data_path, batch_size=64):
    """
    Create dataset iterator only for training and validation set
    Args:
        data_path: image path (string)
        batch_size: number of mini-batch samples
    Returns:
        iterator: iterator initializer
        _next: get next batch
    """
    # convert lists to string tensors
    x_path = glob.glob(data_path + 'images/*.png')
    y_path = [p.replace('images', 'gt') for p in x_path]
    x = tf.constant(x_path, dtype=tf.string)
    y = tf.constant(y_path, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(map_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes)
    _next = iterator.get_next()
    iterator_init = iterator.make_initializer(dataset)
    return iterator_init, _next


def test_iterator(data_path):
    """
    Create dataset iterator only for test set
    Args:
        data_path: image path (string)
    Returns:
        iterator: iterator initializer
        _next: get next batch
    """
    x_path = glob.glob(data_path + 'images/*.png')
    y_path = [p.replace('images', 'pred') for p in x_path]
    x = tf.constant(x_path, dtype=tf.string)
    y = tf.constant(y_path, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(test_map_function)
    iterator = tf.data.Iterator.from_structure(
        dataset.output_types, dataset.output_shapes)
    _next = iterator.get_next()
    iterator_init = iterator.make_initializer(dataset)
    return iterator_init, _next


def map_function(image_path, label_path, crop_height, crop_width, num_batches=64):
    """
    Create transformation function
    Args:
        image_path: image path (string tensor)
        label_path: image path (string tensor)
    Returns:
        images: transformed images
        labels: arrays of segmentation map
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image)
    image = tf.div(tf.cast(image, tf.float32), 255.)

    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label)
    label = tf.squeeze(tf.one_hot(
        tf.div(label, 255), depth=2, dtype=tf.float32))
    # concatinate image  and labels to ensure matching during cropping
    image_label = tf.concat((image, label), axis=-1)
    shape = image_label.get_shape().as_list()
    size = [crop_height, crop_width, shape[-1]]
    def get_patch(x): return tf.stack(
        [tf.image.random_crop(x, size) for _ in range(num_batches)], axis=0)
    batch = get_patch(image_label)
    image, label = batch[:, :, :, :3], batch[:, :, :, 3:]
    return image, label


def test_map_function(image_path, pred_path):
    """
    Create transform function for the test set
    Args:
        image_path: image path (string tensor)
        pred_path: prediction map path (string tensor)
    Returns:
        images: transformed images
        pred_path: string path which is later used to save prediction results
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image)
    image = tf.div(tf.cast(image, tf.float32), 255.)
    # Add batch axis
    image = tf.expand_dims(image, axis=0)
    return image, pred_path


def average_loss(loss):
    """
    Aggregate mini-batch loss statistics and compute average loss
    """
    loss_sum = tf.Variable(initial_value=tf.zeros_like(loss), trainable=False)
    num_batches = tf.Variable(0., trainable=False, dtype=tf.float32)
    accumulate_loss = tf.assign_add(loss_sum, loss)
    accumulate_batch = tf.assign_add(num_batches, 1.)
    ave_loss = loss_sum/num_batches
    update_op = tf.group([accumulate_loss, accumulate_batch])
    reset_op = tf.variables_initializer([loss_sum, num_batches])

    return ave_loss, update_op, reset_op
