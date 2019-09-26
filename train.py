import os
import glob
import warnings
import tensorflow as tf
from hrnet import HRNET
from utils import iterator, average_loss
warnings.filterwarnings('ignore')

KERNEL_SIZE = 3
BATCH_SIZE = 64
NUM_CLASSES = 2
NUM_KERNELS = 8
LR_RATE = 1e-3
HEIGHT = 64
WIDTH = 64
CHANNELS = 3
SAVE_PATH = './model'
DATA_PATH = '../dataset/'
EPOCHS = 100


def train():
    """
    Create training function

    """
    train_file = DATA_PATH + '/train/'
    val_file = DATA_PATH + '/validation/'
    with tf.name_scope('place_holders'):
        x = tf.placeholder(tf.float32, shape=(
            None, None, None, CHANNELS), name='input_image')
        y = tf.placeholder(tf.float32, shape=(
            None, None, None, NUM_CLASSES), name='ground_truth')
        is_train = tf.placeholder(tf.bool, shape=(), name='network_state')

    with tf.name_scope('network'):
        model = HRNET(x, NUM_KERNELS, KERNEL_SIZE, is_train, NUM_CLASSES)
        logits = model.network()
        pred = tf.argmax(tf.nn.softmax(
            logits, axis=-1), axis=-1, output_type=tf.int32)
        print('Output shape: ', logits.shape)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=y))

        ave_loss, loss_update_op, loss_reset_op = average_loss(loss)

    with tf.name_scope('optimizer'):
        optim = tf.train.AdamOptimizer(LR_RATE).minimize(loss)

    with tf.name_scope('performance'):
        mean_iou, mean_iou_update = tf.metrics.mean_iou(labels=tf.argmax(
            y, axis=-1, output_type=tf.int32), predictions=pred,
            num_classes=NUM_CLASSES, name='mean_iou')

        local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        mean_iou_vars = [v for v in local_vars if 'mean_iou' in v.name]

    with tf.name_scope('summary'):
        summary_dir = SAVE_PATH + '/summary/'
        checkpoint_dir = SAVE_PATH + '/model/'
        if not os.path.isdir(summary_dir):
            os.makedirs(summary_dir)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        tf.summary.scalar('validation_mean_iou', mean_iou)
        tf.summary.scalar('validation_loss', ave_loss)
        summary = tf.summary.merge_all()
        summ_writer = tf.summary.FileWriter(summary_dir)
        saver = tf.train.Saver()

    with tf.name_scope('data_pipeline'):
        train_iter, train_next = iterator(train_file, batch_size=BATCH_SIZE)
        val_iter, val_next = iterator(val_file, batch_size=BATCH_SIZE)

    with tf.name_scope('miscellaneous'):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        init_vars = tf.group([tf.global_variables_initializer(),
                              tf.local_variables_initializer()])
        tf.get_default_graph().finalize()

    with tf.Session(config=gpu_config) as sess:
        sess.run(init_vars)
        summ_writer.add_graph(sess.graph)
        print('Training started')
        for i in range(EPOCHS):
            sess.run(train_iter)
            while True:
                try:
                    batch_x, batch_y = sess.run(train_next)
                    sess.run(optim, feed_dict={
                             x: batch_x, y: batch_y, is_train: True})
                except tf.errors.OutOfRangeError:
                    break
            if i % 2 == 0 or i == (EPOCHS - 1):
                sess.run([val_iter, loss_reset_op, mean_iou_vars])
                while True:
                    try:
                        batch_x, batch_y = sess.run(val_next)
                        sess.run([loss_update_op, mean_iou_update], feed_dict={
                                 x: batch_x, y: batch_y, is_train: False})
                    except tf.errors.OutOfRangeError:
                        break
                summ, val_iou, val_loss = sess.run(
                    [summary, mean_iou, ave_loss])
                summ_writer.add_summary(summ, i)
                print('Iteration: {} Mean IoU: {} Loss: {}'.format(
                    i, val_iou, val_loss))
        saver.save(sess, checkpoint_dir + 'hrnet')


if __name__ == "__main__":
    train()
