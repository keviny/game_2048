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
        self._action_number = 4
        self._batch_size = batch_size
        self._eps = 0.8

        # If no input, creates the internal variable, otherwise uses input value.
        self._pl_dict = pl_dict if pl_dict else self.create_placeholder(self._batch_size)
        self._params_dict = params_dict if params_dict else self.create_game_params_dict(self._action_number)
        self._internal_variable = self.generate_model()


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
                                         name='c_b'),
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
        params_dict = {
            'model_conv1_weights': tf.get_variable('model_conv1_weights',
                                                   shape=[4, 4, 1, 128],
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
                                                     [64, action_number],
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

    def generate_model(self):
        internal_variable = {}
        current_q = self.generate_action_q(self.get_pl_dict().get('c_b'))
        internal_variable["current_q"] = current_q
        next_q = self.generate_action_q(self.get_pl_dict().get('n_b'))
        internal_variable["next_q"] = next_q

        internal_variable['c_r'] = self.get_pl_dict().get('c_r')
        internal_variable['c_a'] = self.get_pl_dict().get('c_a')

        # R+r*max(S', a) - Q(S, A) -> 0
        next_score = tf.multiply(self._eps, tf.reduce_max(next_q, axis=1))
        now_score = tf.reduce_sum(tf.multiply(current_q, self.get_pl_dict().get('c_a')), axis=1)
        raw_loss = tf.abs(self.get_pl_dict().get('c_r') + next_score - now_score)

        loss = tf.reduce_mean(raw_loss)
        internal_variable["loss"] = loss
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

        #layer1_output = tf.Print(layer1_output,
        #                         [layer1_output, kernel1, board_pl],
        #                         summarize=16)
        # norm1, output size is same as input
        # http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
        # norm1 = tf.nn.lrn(layer1_output, 2, bias=1.0, alpha=0.1 / 9.0, beta=0.75, name='norm1')

        # conv2, output is [batch_size, 4, 4, 64]
        with tf.variable_scope('%s_conv2' % self._name) as scope:
            kernel2 = self._params_dict.get('model_conv2_weights')
            conv2 = tf.nn.conv2d(layer1_output, kernel2, [1, 1, 1, 1], padding='SAME')
            biases2 = self._params_dict.get('model_conv2_biases')
            sum_value2 = tf.nn.bias_add(conv2, biases2)
            layer2_output = tf.nn.relu(sum_value2, "layer2_output")

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
        with tf.variable_scope('%s_local3' % self._name) as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape3 = tf.reshape(pool2, [self._batch_size, 4 * 64])
            dropout_reshape3 = tf.nn.dropout(reshape3, dropout_keep_prob)
            weights3 = self._params_dict.get('model_local3_weights')
            biases3 = self._params_dict.get('model_local3_biases')
            matmul_value = tf.matmul(dropout_reshape3, weights3) + biases3
            layer3_output = tf.nn.relu(matmul_value, name="layer3_output")

        # matmul4
        # weight is [128, 4], output is [batch_size, 4]
        with tf.variable_scope('%s_matmul4' % self._name) as scope:
            weights4 = self._params_dict.get('model_matmul4_weights')
            biases4 = self._params_dict.get('model_matmul4_biases')
            layer4_output = tf.add(tf.matmul(layer3_output, weights4), biases4, name="layer4_output")
            # expected_q = tf.nn.softmax(tf.matmul(layer3_output, weights4) + biases4)
            # _activation_summary(softmax_linear)
        # Save the model in a specific name.
        action_q = layer4_output
        # output the final data, [batch_size, ACTION_NUMBER]
        # The first return value is useful, the second and third is for debugging.
        # params_dict.get('model_conv1_weights')
        # expected_q = tf.Print(expected_q,[expected_q, layer3_output, layer2_output, layer1_output],'debug info: ', summarize=16)
        # expected_q = tf.Print(expected_q, [expected_q, params_dict.get('model_conv1_weights'), params_dict.get('model_conv1_biases')], "debug info:", summarize=16)
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
    tensor = gm.get_internal_variable().get("loss")
    tensor1 = gm.get_internal_variable().get("current_q")
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
        value = sess.run([tensor, tensor1],
                         feed_dict={gm.get_pl_dict().get('c_b'): np.array(target_board).reshape(2, 4, 4, 1).astype(float),
                                    gm.get_pl_dict().get('n_b'): np.array(next_board).reshape(2, 4, 4, 1).astype(float),
                                    gm.get_pl_dict().get('c_r'): np.array(rev).reshape(2, 1).astype(float),
                                    gm.get_pl_dict().get('c_a'): np.array(target_action).reshape(2, 4).astype(float),
                                    gm.get_pl_dict().get('dropout_keep_prob'): 0.3})
        print value[0]
        print value[1]


