import math
import game_pool

import tensorflow as tf
import numpy as np


class GameModel(object):

    """
    Constructor of GameModel object.

    Args:
      batch_size is the size of row will be processed in batch.
      action_number is the number of output in the model.
    """
    def __init__(self, batch_size, pl_dict=None, params_dict=None):
        self._action_number = 4
        self._memory_length = 4
        self._batch_size = batch_size
        # If no input, creates the internal variable, otherwise uses input value.
        self._pl_dict = pl_dict if pl_dict else GameModel.create_placeholder(batch_size)
        self._params_dict = params_dict if params_dict else GameModel.create_game_params_dict(self._action_number)
        self._internal_variable = []

    @staticmethod
    def create_placeholder(batch_size):
        """Creates the pl dict needed for the computing graph.

        Args:
          batch_size: the size of each batch.

        Returns:
          The pl dict includes current board and score,
          and boards and scores tuple list for 4 directions.
        """
        # c_b -> current board, (batch_size, high, width, snapshot)
        # c_m -> current max, (batch_size, snapshot)
        # c_r -> current revenue, how much revenue got after this move.
        # c_a -> current action, the current movement direction.
        # dropout_keep_prod -> dropout keep probability.
        pl_dict = {'c_b': tf.placeholder(tf.float32,
                                         shape=(batch_size, 4, 4, 4),
                                         name='c_b'),
                   'c_m': tf.placeholder(tf.float32,
                                         shape=(batch_size, 4),
                                         name='c_m'),
                   'c_r': tf.placeholder(tf.float32,
                                         shape=(batch_size, 1),
                                         name='c_r'),
                   'c_a': tf.placeholder(tf.float32,
                                         shape=(batch_size, 4),
                                         name='c_a'),
                   'dropout_keep_prob': tf.placeholder(tf.float32)}
        return pl_dict

    @staticmethod
    def create_game_params_dict(action_number):
        """Generate the model parameters in the Q NN.

        Those parameters will be shared to calculate the Q value.
        """
        params_dict = {
            'model_conv1_weights': tf.get_variable('model_conv1_weights',
                                                   shape=[4, 4, 4, 128],
                                                   initializer=tf.truncated_normal_initializer(
                                                       mean=0.0, stddev=0.01)),
            'model_conv1_biases': tf.get_variable('model_conv1_biases',
                                                  shape=[128],
                                                  initializer=tf.truncated_normal_initializer(
                                                      mean=0.0, stddev=0.01)),
            'model_conv2_weights': tf.get_variable('model_conv2_weights',
                                                   shape=[4, 4, 128, 64],
                                                   initializer=tf.truncated_normal_initializer(
                                                       mean=0.0, stddev=0.01)),
            'model_conv2_biases': tf.get_variable('model_conv2_biases',
                                                  shape=[64],
                                                  initializer=tf.truncated_normal_initializer(
                                                      mean=0.0, stddev=0.01)),
            'model_local3_weights': tf.get_variable('model_local3_weights',
                                                    shape=[64 * 4, 64],
                                                    initializer=tf.truncated_normal_initializer(
                                                        mean=0.0, stddev=0.01)),
            'model_local3_biases': tf.get_variable('model_local3_biases',
                                                   shape=[64],
                                                   initializer=tf.truncated_normal_initializer(
                                                       mean=0.0, stddev=0.01)),
            # Below is the output layer, the size should be decided by the number of action.
            'model_matmul4_weights': tf.get_variable('model_matmul4_weights',
                                                     [68, action_number],
                                                     tf.float32,
                                                     initializer=tf.truncated_normal_initializer(
                                                         mean=0.0, stddev=0.01)),
            'model_matmul4_biases': tf.get_variable('model_matmul4_biases',
                                                    [action_number],
                                                    tf.float32,
                                                    initializer=tf.truncated_normal_initializer(
                                                         mean=0.0, stddev=1/32.0))
        }
        return params_dict

    def get_unifrom_regularization(self):
        regularization_list = []
        for pl in self._params_dict.values():
            regularization_list.append(tf.sqrt(tf.reduce_sum(tf.square(pl))))
        return sum(regularization_list)

    def get_classified_regularization(self):
        regularization_list = [
            self.get_classified_regularization_item('model_conv1_weights'),
            self.get_classified_regularization_item('model_conv1_biases'),
            self.get_classified_regularization_item('model_conv2_weights'),
            self.get_classified_regularization_item('model_conv2_biases'),
            self.get_classified_regularization_item('model_local3_weights'),
            self.get_classified_regularization_item('model_local3_biases'),
            self.get_classified_regularization_item('model_matmul4_weights'),
            self.get_classified_regularization_item('model_matmul4_biases')]
        return sum(regularization_list)

    def get_classified_regularization_item(self, name):
        """ Get the classified regularization for one parameter. """
        pl = self._params_dict[name]
        return tf.sqrt(tf.reduce_sum(tf.square(pl))) / float(sum(pl.get_shape().as_list()))

    def evaluate_target_board(self, target_pl_dict, params_dict):
        """Generates the reinforcement learning loss function"""
        current_q, _, _ = generate_game_q(target_pl_dict, params_dict, 1, 'evaluate_board')
        return current_q

    def generate_action_q(self, name="board"):
        """
        Use the game board as the input. The input is [batch_size, 4, 4, 1]
        Args:
            batch_size: the size of each batch.
        Returns:
            The q value of the current board, and two extra variable for debuging purpose.
            debug 1 variable
            debug 2 variable
        """
        # Hidden 1
        # input is [batch_size, 4, 4, 1] * [4, 4, 4, 128]
        # output is [batch_size, 4, 4, 128]
        # This model need two placeholders
        #  one is current board(c_b) and dropout keep prob.
        board_pl = self._pl_dict['c_b']
        max_pl = self._pl_dict['c_m']
        dropout_keep_prob = self._pl_dict['dropout_keep_prob']

        with tf.variable_scope('%s_conv1' % name) as scope:
            kernel1 = self._params_dict.get('model_conv1_weights')
            conv1 = tf.nn.conv2d(board_pl, kernel1, [1, 1, 1, 1], padding='SAME')
            biases1 = self._params_dict.get('model_conv1_biases')
            sum_value1 = tf.nn.bias_add(conv1, biases1)
            layer1_output = tf.nn.relu(sum_value1)
            self._internal_variable.append(conv1)

        #layer1_output = tf.Print(layer1_output,
        #                         [layer1_output, kernel1, board_pl],
        #                         summarize=16)
        # norm1, output size is same as input
        # http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
        # norm1 = tf.nn.lrn(layer1_output, 2, bias=1.0, alpha=0.1 / 9.0, beta=0.75, name='norm1')

        # conv2, output is [batch_size, 4, 4, 64]
        with tf.variable_scope('%s_conv2' % name) as scope:
            kernel2 = self._params_dict.get('model_conv2_weights')
            conv2 = tf.nn.conv2d(layer1_output, kernel2, [1, 1, 1, 1], padding='SAME')
            biases2 = self._params_dict.get('model_conv2_biases')
            sum_value2 = tf.nn.bias_add(conv2, biases2)
            layer2_output = tf.nn.relu(sum_value2)
            self._internal_variable.append(conv2)

        # The output size is same as input. [batch_size, 4, 4, 128]
        # norm2 = tf.nn.lrn(layer2_output, 2, bias=1.0, alpha=0.1 / 9.0, beta=0.75, name='norm2')

        # pool2
        # output is [batch_size, 2, 2, 64]
        pool2 = tf.nn.max_pool(layer2_output,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # local3
        # weight is [128 * 4, 128] output is [batch_size, 128]
        with tf.variable_scope('%s_local3' % name) as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape3 = tf.reshape(pool2, [self._batch_size, 4 * 64])
            dropout_reshape3 = tf.nn.dropout(reshape3, dropout_keep_prob)
            weights3 = self._params_dict.get('model_local3_weights')
            biases3 = self._params_dict.get('model_local3_biases')
            matmul_value = tf.matmul(dropout_reshape3, weights3) + biases3
            layer3_rule = tf.nn.relu(matmul_value)
            layer3_output = tf.concat(axis=1, values=[layer3_rule, max_pl])
            self._internal_variable.append(layer3_rule)


        # matmul4
        # weight is [128, 4], output is [batch_size, 4]
        with tf.variable_scope('%s_matmul4' % name) as scope:
            weights4 = self._params_dict.get('model_matmul4_weights')
            biases4 = self._params_dict.get('model_matmul4_biases')
            expected_q = tf.matmul(layer3_output, weights4) + biases4
            self._internal_variable.append(expected_q)
            # expected_q = tf.nn.softmax(tf.matmul(layer3_output, weights4) + biases4)
            # _activation_summary(softmax_linear)

        # output the final data, [batch_size, ACTION_NUMBER]
        # The first return value is useful, the second and third is for debugging.
        # params_dict.get('model_conv1_weights')
        # expected_q = tf.Print(expected_q,[expected_q, layer3_output, layer2_output, layer1_output],'debug info: ', summarize=16)
        # expected_q = tf.Print(expected_q, [expected_q, params_dict.get('model_conv1_weights'), params_dict.get('model_conv1_biases')], "debug info:", summarize=16)
        return expected_q, max_pl, layer3_output

    def get_pl_dict(self):
      return self._pl_dict

    def get_params_dict(self):
      return self._params_dict

    def get_internal_variable(self):
      return self._internal_variable

if __name__ == "__main__":
    gm = GameModel(2)
    current_q, conv1, conv2 = gm.generate_action_q('evaluate_board')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        print init
        sess.run(init)
        target_board = [32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 1,
                        32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 2,
                        32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 3,
                        32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 4,
                        32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 5,
                        32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 6,
                        32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 7,
                        32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 8]
        target_max = [2,3,4,5,6,7,8,9]
        value = sess.run([current_q, conv1, conv2],
            feed_dict={gm.get_pl_dict().get('c_b'): np.array(target_board).reshape(2, 4, 4, 4).astype(float),
                       gm.get_pl_dict().get('c_m'): np.array(target_max).reshape(2,4).astype(float),
                       gm.get_pl_dict().get('dropout_keep_prob'): 0.3})
        print "q value:%s" % value[0]
        print "extra1: %s" % value[1]
        import pdb
        pdb.set_trace()
        print "extra2: %s" % value[2]


