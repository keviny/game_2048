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
    def __init__(self, name, batch_size, pl_dict=None, params_dict=None):
        self._name = name
        self._action_number = 1
        self._batch_size = batch_size
        self._eps = 0.9

        # If no input, creates the internal variable, otherwise uses input value.
        self._pl_dict = pl_dict if pl_dict else self.create_placeholder(self._batch_size)
        self._params_dict = params_dict if params_dict else self.create_game_params_dict(self._action_number)
        self._internal_variable = self.generate_model()
        self._name = "model_s"

    def get_name(self):
        return self._name;

    def create_placeholder(self, batch_size):
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
                                         shape=(batch_size, 4, 4, 1),
                                         name='c_b'),
                   'n_b': tf.placeholder(tf.float32,
                                         shape=(batch_size, 4, 4, 1),
                                         name='n_b'),
                   'c_r': tf.placeholder(tf.float32,
                                         shape=(batch_size, 1),
                                         name='c_r'),
                   'c_a': tf.placeholder(tf.float32,
                                         shape=(batch_size, 4),
                                         name='c_a'),
                   'dropout_keep_prob': tf.placeholder(tf.float32)}
        return pl_dict

    def create_game_params_dict(self, action_number):
        """Generate the model parameters in the Q NN.

        Those parameters will be shared to calculate the Q value.
        """
        layer1_output = 128
        layer2_output = 64
        """Generate the model parameters in the Q NN.
        Those parameters will be shared to calculate the Q value.
        """
        params_dict = {
            'model_conv1_weights': tf.get_variable('model_conv1_weights',
                                                   shape=[4, 4, 1, layer1_output],
                                                   initializer=tf.truncated_normal_initializer(
                                                       mean=0.0, stddev=0.01)),
            'model_conv1_biases': tf.get_variable('model_conv1_biases',
                                                  shape=[layer1_output],
                                                  initializer=tf.truncated_normal_initializer(
                                                      mean=0.0, stddev=0.01)),
            'model_conv2_weights': tf.get_variable('model_conv2_weights',
                                                   shape=[4, 4, layer1_output, layer2_output],
                                                   initializer=tf.truncated_normal_initializer(
                                                       mean=0.0, stddev=0.01)),
            'model_conv2_biases': tf.get_variable('model_conv2_biases',
                                                  shape=[layer2_output],
                                                  initializer=tf.truncated_normal_initializer(
                                                      mean=0.0, stddev=0.01)),
            # Below is the output layer, the size should be decided by the number of action.
            'model_matmul4_weights': tf.get_variable('model_matmul4_weights',
                                                     [layer2_output * 16, action_number],
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

    def evaluate_target_board(self, target_pl_dict, params_dict):
        """Generates the reinforcement learning loss function"""
        current_q, _, _ = generate_game_q(target_pl_dict, params_dict, 1, 'evaluate_board')
        return current_q

    def generate_model(self):
        internal_variable = {}
        current_q = self.generate_action_q(self.get_pl_dict().get('c_b'))
        internal_variable["current_q"] = current_q
        next_q = self.generate_action_q(self.get_pl_dict().get('n_b'))
        internal_variable["next_q"] = next_q

        internal_variable['c_r'] = self.get_pl_dict().get('c_r')
        internal_variable['c_a'] = self.get_pl_dict().get('c_a')

        # R + Q(S+1) - Q(S) -> 0
        # next_score = tf.multiply(self._eps, tf.reduce_max(next_q, axis=1))
        raw_loss = tf.abs(self.get_pl_dict().get('c_a') - current_q + tf.multiply(self._eps, next_q))

        internal_variable["raw_loss"] = raw_loss
        return internal_variable

    def generate_action_q(self, board_pl):
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
        dropout_keep_prob = self._pl_dict['dropout_keep_prob']
        internal_variable = {}

        with tf.variable_scope('%s_conv1' % self._name) as scope:
            kernel1 = self._params_dict.get('model_conv1_weights')
            conv1 = tf.nn.conv2d(board_pl, kernel1, [1, 1, 1, 1], padding='SAME')
            biases1 = self._params_dict.get('model_conv1_biases')
            sum_value1 = tf.nn.bias_add(conv1, biases1)
            layer1_output = tf.nn.relu(sum_value1, "layer1_output")

         # conv2, output is [batch_size, 4, 4, 64]
        with tf.variable_scope('%s_conv2' % self._name) as scope:
            kernel2 = self._params_dict.get('model_conv2_weights')
            conv2 = tf.nn.conv2d(layer1_output, kernel2, [1, 1, 1, 1], padding='SAME')
            biases2 = self._params_dict.get('model_conv2_biases')
            sum_value2 = tf.nn.bias_add(conv2, biases2)
            layer2_output = tf.nn.relu(sum_value2, "layer2_output")

        # matmul4
        # weight is [128, 4], output is [batch_size, 4]
        with tf.variable_scope('%s_matmul4' % self._name) as scope:
            weights4 = self._params_dict.get('model_matmul4_weights')
            biases4 = self._params_dict.get('model_matmul4_biases')
            layer2_output_reshape = tf.reshape(layer2_output, [self._batch_size, 64 * 16])
            layer4_output = tf.add(tf.matmul(layer2_output_reshape, weights4), biases4, name="layer4_output")

        action_q = layer4_output

        return action_q

    def create_feed_dict(self, batch_size, current_board, next_board, current_action, current_revenue):
        feed_dict = {}

        if current_board is not None:
            feed_dict[self._pl_dict['c_b']] = np.array(current_board).reshape(batch_size, 4, 4, 1).astype(float)
        if next_board is not None:
            feed_dict[self._pl_dict['n_b']] = np.array(next_board).reshape(batch_size, 4, 4, 1).astype(float)
        if current_action is not None:
            feed_dict[self._pl_dict['c_a']] = np.array(current_action).reshape(batch_size, 4).astype(float)
        if current_revenue is not None:
            feed_dict[self._pl_dict['c_r']] = np.array(current_revenue).reshape(batch_size, 1).astype(float)
        feed_dict[self._pl_dict['dropout_keep_prob']] = 1.0
        return feed_dict

    def get_pl_dict(self):
        return self._pl_dict

    def get_params_dict(self):
        return self._params_dict

    def get_batch_size(self):
        return self._batch_size;

    def get_internal_variable(self):
        return self._internal_variable

if __name__ == "__main__":
    gm = GameModel("train", 2)
    gm.generate_model()
    tensor = gm.get_internal_variable().get("raw_loss")
    tensor1 = gm.get_internal_variable().get("current_q")
    print tensor
    print tensor1
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        print init
        sess.run(init)
        target_board = [32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 1,
                        32, 8, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 2]
        next_board = [32, 1, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 1,
                        32, 1, 2, 2, 4, 256, 8, 2, 2, 4, 32, 4, 4, 32, 8, 2]
        target_action = [[0, 0, 1, 0], [1, 0, 0, 0]]
        rev = [0, 1]
        print gm.get_pl_dict()
        feed_dict=gm.create_feed_dict(
            gm.get_batch_size(),
            target_board,
            next_board,
            target_action,
            rev)
        print feed_dict
        value = sess.run([tensor, tensor1],
                         feed_dict=feed_dict)
        print value[0]
        print value[1]


