import os
import glob
import tensorflow as tf
import numpy as np
from hrnet import HRNET
from utils import iterator, average_loss, test_iterator

KERNEL_SIZE = 3
BATCH_SIZE = 32
NUM_CLASSES = 2
NUM_KERNELS = 32
LR_RATE = 1e-3
CHANNELS = 3
SAVE_PATH = './HRNET_32'
DATA_PATH = '../dataset/'
EPOCHS = 100


def train():
    """
    Create training function

    """
    train_file = DATA_PATH + '/train/'
    val_file = DATA_PATH + '/validation/'
    test_file = DATA_PATH + '/test/'
    if not os.path.isdir(test_file + 'pred'):
        os.makedirs(test_file + 'pred')
    tf.reset_default_graph()
    with tf.name_scope('place_holders'):
        x = tf.placeholder(tf.float32, shape=(
            None, None, None, CHANNELS), name='input_image')
        y = tf.placeholder(tf.float32, shape=(
            None, None, None, NUM_CLASSES), name='ground_truth')
        is_train = tf.placeholder(tf.bool, shape=(), name='network_state')

    with tf.name_scope('network'):
        model = HRNET(x, NUM_KERNELS, KERNEL_SIZE, is_train, NUM_CLASSES)
        logits = model.network()
        # generate segmentation map
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
        test_iter, test_next = test_iterator(test_file)

    with tf.name_scope('miscellaneous'):
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        init_vars = tf.group([tf.global_variables_initializer(),
                              tf.local_variables_initializer()])
        tf.get_default_graph().finalize()

    with tf.Session(config=gpu_config) as sess:
        sess.run(init_vars)
        summ_writer.add_graph(sess.graph)
        print('Training started....')
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
            if i % 25 == 1 or i == (EPOCHS - 1):
                saver.save(sess, checkpoint_dir + 'hrnet', global_step=i+1)
        print('Finished training.....')
        ############################################
        # GENERATE PREDICTION MAPS FOR TEST IMAGES##
        ############################################
        sess.run(test_iter)
        while True:
            try:
                test_x, save_path = sess.run(test_next)
                def seg_map_batch(im): return sess.run(
                    pred, feed_dict={x: im, is_train: False})
                seg_map = [seg_map_batch(np.expand_dims(x, axis=0))
                           for x in test_x]
                seg_map = np.concatenate(seg_map, axis=0)
                # save segmentation map
                save_seg_maps(np.squeeze(seg_map), save_path.decode('utf-8'))
            except tf.errors.OutOfRangeError:
                print('Finished test evaluation')
                break


if __name__ == "__main__":
    train()
