"""Tensorflow local version.

To start the tensorboard, run:
learning/brain/tensorboard/tensorboard.sh --port 2222 --logdir /tmp/sug_logs
"""

import game_model
import game_pool
import tensorflow as tf
import logging
import logging.handlers
import time
from tensorflow.python.training.summary_io import SummaryWriter

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('output_path', '/tmp/log/',
                    'Tensorflow log will be under /tmp/[output_path]')
flags.DEFINE_string('exp_name', 'test1', 'The experiment name')
flags.DEFINE_integer('generator_type', 2, 'Generator type, 1:Model, 2:Random(default)')

logger = logging.getLogger(FLAGS.exp_name)

handler = logging.handlers.RotatingFileHandler(
    '%s/log/%s.log' % (FLAGS.output_path, FLAGS.exp_name),
    maxBytes = 1024*1024,
    backupCount = 5)

fmt = '%(asctime)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger.addHandler(handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)

logger.setLevel(logging.DEBUG)


def scalar_summary_detail(scalar_name, scalar_tensor, averages):
    """Record the scalar op value, which need to be triggered."""
    tf.summary.scalar(scalar_name, scalar_tensor)
    scalar_tensor_log = tf.log(scalar_tensor)
    tf.summary.scalar(scalar_name + '_log', scalar_tensor_log)
    averages = tf.train.ExponentialMovingAverage(0.95)
    average_op = averages.apply([scalar_tensor_log])
    tf.summary.scalar(scalar_name + '_log_average',
                      averages.average(scalar_tensor_log))
    # Please note, this op must be triggered explicitly.
    return average_op


def train():
    epoch_size = 32
    batch_size = 128
    decay_iteration = 10000
    total_iteration = decay_iteration * 8             # 8,000,000

    """Trains the game_model."""
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        game_obj = game_model.GameModel("train", batch_size)
        global_step = tf.Variable(0, trainable=False)
        averages = tf.train.ExponentialMovingAverage(0.95, global_step)

        # Get the score of each direction.
        actions_q = game_obj.get_internal_variable().get("action_q")
        # Only for test purpose.
        regularization = game_obj.get_classified_regularization()
        pl_dict = game_obj.get_pl_dict()
        # pl_dict['c_a'] -> (batch_size, 4)
        # action_q -> (batch_size, 4)
        # The output of multiply will be (batch_size, 4), reduce_sum will output (batch_size, 4)
        expacted_reward = tf.reduce_sum(tf.multiply(actions_q, pl_dict['c_a']), axis=1)
        # Generate the sum of average diff and plus the reg to calculate the epoch loss.
        epoch_loss = tf.reduce_mean(tf.abs(expacted_reward - pl_dict['c_r'])) + regularization * 0.1;
        epoch_loss_op = scalar_summary_detail('epoch_loss', epoch_loss, averages)

        # Track the moving averages of all trainable variables.
        # variables_averages_op = averages.apply(tf.trainable_variables())

        # Explicitly trigger waiting for the average loss tensor.
        with tf.control_dependencies([epoch_loss_op]), tf.name_scope('update'):
            learning_rate = tf.train.exponential_decay(
                0.3, global_step, epoch_size * decay_iteration, 0.5, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            grads = opt.compute_gradients(epoch_loss)
            train_op = opt.apply_gradients(grads, global_step=global_step)

        # Record all summary data.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        for var in game_obj.get_internal_variable().values():
            tf.summary.histogram(var.op.name, var)

        tf.summary.merge_all()
        summary_op = tf.summary.merge_all()
        summary_writer = SummaryWriter('%s/summary/%s/' % (FLAGS.output_path, FLAGS.exp_name), sess.graph)
        saver = tf.train.Saver(max_to_keep=100)
        init = tf.global_variables_initializer()
        logger.info(init)
        logger.info("model ready")

        # Creates the game object, using the existing model.
        eval_game_obj = game_model.GameModel("eval", 1, params_dict=game_obj.get_params_dict())
        pool = game_pool.GamePool(500000, sess, eval_game_obj)
        logger.info("gen graph ready")
        sess.run(init)
        stat_info = pool.get_stat_info()
        pool.generate_training_data()
        for i in xrange(total_iteration):
            total_loss_value = 0.0
            start_time = time.time()
            pool.generate_training_data()
            for p in xrange(epoch_size):
                feed_dict = pool.create_external_feed_dict(game_obj.get_pl_dict(), game_obj.get_batch_size())
                # Core training function.
                _, epoch_loss_value, learning_rate_value = \
                    sess.run([train_op, epoch_loss, learning_rate],
                             feed_dict=feed_dict)
                total_loss_value = total_loss_value + epoch_loss_value
            logger.info('time:%.3fs e_s:%d, e_n:%d lr:%.5f average_loss:%.3f' % (
                 time.time() - start_time,
                 epoch_size,
                 i,
                 learning_rate_value,
                 total_loss_value / float(epoch_size)))

            if i % 300 == 0:
                saver.export_meta_graph('%s/checkpoint/%s/model.meta' % (FLAGS.output_path, FLAGS.exp_name))
                saver.save(sess,
                           '%s/checkpoint/%s/model.ckpt' % (FLAGS.output_path, FLAGS.exp_name),
                           global_step=global_step)
                logger.info("Training model saved")
            if i % 10 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()
                stat_info = pool.get_stat_info_string()
                logger.info('[%s] Summary saved. %s' % (FLAGS.exp_name, stat_info))
        logger.info('[%s] Final stat_info:%s' % (FLAGS.exp_name, stat_info))


def main(unused_argv):
    """ Generate some debug info and start trainning. """
    logger.info('exp_name:%s, generator_type:%d' % (FLAGS.exp_name, FLAGS.generator_type))
    logger.info('output_path:%s' % FLAGS.output_path)
    train()


if __name__ == '__main__':
    tf.app.run()
