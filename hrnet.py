import tensorflow as tf
from utils import resnet_unit, conv2d, batch_norm, lrelu, upsample


class HRNET:
    def __init__(self, x, num_kernels, kernel_size, is_train, num_classes):
        """
        Init function
        """
        self.x = x
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.is_train = is_train
        self.num_classes = num_classes

    def network(self):
        """
        Create the HRNET network
        """
        conv_input = conv2d(self.x, self.num_kernels,
                            self.kernel_size, var_scope='conv_input')
        bn = batch_norm(conv_input, self.is_train, var_scope='bn_input')
        act = lrelu(bn)
        branch_one_resnet_one = resnet_unit(
            act, self.num_kernels, self.is_train, var_scope='branch_one_resnet_one')

        branch_one_resnet_two = resnet_unit(
            branch_one_resnet_one, 2*self.num_kernels, self.is_train,
            var_scope='branch_one_resnet_two')

        branch_two_resnet_one = resnet_unit(
            branch_one_resnet_one, 2*self.num_kernels, self.is_train, stride=2,
            var_scope='branch_two_resnet_one')

        branch_one_upsample_shape = tf.shape(self.x)[1:3]
        branch_two_resnet_one_upsample = upsample(
            branch_two_resnet_one, branch_one_upsample_shape)
        branch_one_resnet_three_input = tf.add(
            branch_one_resnet_two, branch_two_resnet_one_upsample)

        branch_one_resnet_three = resnet_unit(
            branch_one_resnet_three_input, self.num_kernels, self.is_train,
            var_scope='branch_one_resnet_three')

        branch_two_conv_one = conv2d(
            branch_one_resnet_two, 2*self.num_kernels, 3, stride=2,
            var_scope='branch_two_conv_one')
        branch_two_bn_one = batch_norm(
            branch_two_conv_one, self.is_train, var_scope='branch_two_bn_one')
        branch_two_act_one = lrelu(branch_two_bn_one)

        branch_two_resnet_two_input = tf.add(
            branch_two_resnet_one, branch_two_act_one)

        branch_two_resnet_two = resnet_unit(
            branch_two_resnet_two_input, 2*self.num_kernels, self.is_train,
            var_scope='branch_two_resnet_two')

        branch_three_resnet_one = resnet_unit(
            branch_two_resnet_two_input, 4*self.num_kernels, self.is_train, stride=2,
            var_scope='branch_three_resnet_two_input')

        branch_two_resnet_two_upsample = upsample(
            branch_two_resnet_two, branch_one_upsample_shape)
        branch_three_resnet_one_upsample = upsample(
            branch_three_resnet_one, branch_one_upsample_shape)

        multi_res_concat = tf.concat(
            (branch_one_resnet_three, branch_two_resnet_two_upsample, branch_three_resnet_one_upsample), axis=-1)

        print('Multi resolution concat shape: ', multi_res_concat.shape)

        output_conv = conv2d(
            multi_res_concat, self.num_classes, self.kernel_size,
            var_scope='output_conv')
        return output_conv
