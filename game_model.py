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
        self._name = "model"

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
        params_dict = {
            'layer1_1_2_conv_weights': tf.get_variable('layer1_1_2_conv_weights',
                                                       shape=[1, 2, 1, layer1_output],
                                                       initializer=tf.truncated_normal_initializer(
                                                          mean=0.0, stddev=0.01)),
            'layer1_2_1_conv_weights': tf.get_variable('layer1_2_1_conv_weights',
                                                       shape=[2, 1, 1, layer1_output],
                                                       initializer=tf.truncated_normal_initializer(
                                                          mean=0.0, stddev=0.01)),
            'layer2_1_2_1_2_conv_weights': tf.get_variable('layer2_1_2_1_2_conv_weights',
                                                       shape=[1, 2, layer1_output, layer2_output],
                                                       initializer=tf.truncated_normal_initializer(
                                                           mean=0.0, stddev=0.01)),
            'layer2_1_2_2_1_conv_weights': tf.get_variable('layer2_1_2_2_1_conv_weights',
                                                       shape=[2, 1, layer1_output, layer2_output],
                                                       initializer=tf.truncated_normal_initializer(
                                                           mean=0.0, stddev=0.01)),
            'layer2_2_1_1_2_conv_weights': tf.get_variable('layer2_2_1_1_2_conv_weights',
                                                       shape=[1, 2, layer1_output, layer2_output],
                                                       initializer=tf.truncated_normal_initializer(
                                                           mean=0.0, stddev=0.01)),
            'layer2_2_1_2_1_conv_weights': tf.get_variable('layer2_2_1_2_1_conv_weights',
                                                       shape=[2, 1, layer1_output, layer2_output],
                                                       initializer=tf.truncated_normal_initializer(
                                                           mean=0.0, stddev=0.01)),
            # Below is the output layer, the size should be decided by the number of action.
            'model_matmul4_weights': tf.get_variable('model_matmul4_weights',
                                                     [layer2_output * (8 + 9) * 2 + layer1_output * (12 + 12), action_number],
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

        # R + Q(S+1) - Q(S) -> 0
        # next_score = tf.multiply(self._eps, tf.reduce_max(next_q, axis=1))
        raw_loss = tf.abs(self.get_pl_dict().get('c_r') - current_q + tf.multiply(self._eps, next_q))
        internal_variable["c_r"] = self.get_pl_dict().get('c_r');
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

        layer1_1_2_conv_weights = self._params_dict.get('layer1_1_2_conv_weights')
        layer1_2_1_conv_weights = self._params_dict.get('layer1_2_1_conv_weights')

        layer2_1_2_1_2_conv_weights = self._params_dict.get('layer2_1_2_1_2_conv_weights')
        layer2_1_2_2_1_conv_weights = self._params_dict.get('layer2_1_2_2_1_conv_weights')

        layer2_2_1_1_2_conv_weights = self._params_dict.get('layer2_2_1_1_2_conv_weights')
        layer2_2_1_2_1_conv_weights = self._params_dict.get('layer2_2_1_2_1_conv_weights')

        model_matmul4_weights = self._params_dict.get('model_matmul4_weights')
        model_matmul4_biases = self._params_dict.get('model_matmul4_biases')

        conv1_1_2 = tf.nn.conv2d(board_pl, layer1_1_2_conv_weights, strides=[1, 1, 1, 1], padding='VALID', name='conv1_1_2')
        conv1_2_1 = tf.nn.conv2d(board_pl, layer1_2_1_conv_weights, strides=[1, 1, 1, 1], padding='VALID', name='conv1_2_1')

        conv1_1_2_1_2 = tf.nn.conv2d(conv1_1_2, layer2_1_2_1_2_conv_weights, strides=[1, 1, 1, 1], padding='VALID')
        conv1_1_2_2_1 = tf.nn.conv2d(conv1_1_2, layer2_1_2_2_1_conv_weights, strides=[1, 1, 1, 1], padding='VALID')

        conv1_2_1_1_2 = tf.nn.conv2d(conv1_2_1, layer2_2_1_1_2_conv_weights, strides=[1, 1, 1, 1], padding='VALID')
        conv1_2_1_2_1 = tf.nn.conv2d(conv1_2_1, layer2_2_1_2_1_conv_weights, strides=[1, 1, 1, 1], padding='VALID')

        concated_input = tf.concat([
          tf.reshape(conv1_1_2, [self._batch_size, -1]),
          tf.reshape(conv1_2_1, [self._batch_size, -1]),
          tf.reshape(conv1_1_2_1_2, [self._batch_size, -1]),
          tf.reshape(conv1_1_2_2_1, [self._batch_size, -1]),
          tf.reshape(conv1_2_1_1_2, [self._batch_size, -1]),
          tf.reshape(conv1_2_1_2_1, [self._batch_size, -1])], 1)

        conv_input = tf.reshape(concated_input, [self._batch_size, -1])

        layer4_output = tf.add(tf.matmul(conv_input, model_matmul4_weights), model_matmul4_biases, name="layer4_output")
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
    tensor2 = gm.get_internal_variable().get("next_q")
    tensor3 = gm.get_internal_variable().get("c_r")
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
        value = sess.run([tensor, tensor1, tensor2, tensor3],
                         feed_dict=feed_dict)
        print "raw_loss"
        print value[0]
        print "current_q"
        print value[1]
        print "next_q"
        print value[2]
        print "c_r"
        print value[3]


