import time
import game
import random
import numpy as np
from collections import deque


class GamePool(object):
    MODEL_GENERATOR = 1
    RANDOM_GENERATOR = 2

    def __init__(self, pool_size, sess, pl_dict, actions_q, batch_size):
        """
        The training data pool is a collection of each training record.
        Each training record is composed of (s0, a0, r0, s1)
        s0, s1 represents the current board status and next step board status.
        a0 represents the action executed in s0
        r0 represents the incremental score of this execution

        Args:
          pool_size: The number of record this pool could contain
          sess: Session variable.
          pl_dict: MODEL_GENERATOR (1) or RANDOM_GENERATOR (2)
          actions_q: The q value tensor, which will be used if we need to generate the training data by ML model.
          batch_size: The number of record of each batch
        """
        self._pl_board = pl_dict['c_b']
        self._dropout_keep_prob = pl_dict['dropout_keep_prob']
        self._pl_max = pl_dict['c_m']
        self._sess = sess
        self._batch_size = batch_size
        self._actions_q = actions_q
        self._counter = 0
        self._hold_step = 10

        # Build the internal objects and start to load the record.
        self._training_node_pool = deque(maxlen=pool_size)

        # For stats, just collect the latest 512 items.
        self._game_score_pool = deque(maxlen=512)
        self._game_step_pool = deque(maxlen=512)
        self._game_beyond_pool = deque(maxlen=512)

        # The internal game object used to generate the training data.
        self._game_obj = game.Game()

        random.seed(time.time() * 1000)

    def generate_training_data(self):
        """Loads the auto generated training records into the pool. """

        # Reset the game if needed.
        self._game_obj.reset()
        current_max_element = float(max(self._game_obj.get_board()))
        current_board = np.array(self._game_obj.get_board()) / current_max_element
        current_max_element = float(max(current_board))
        current_board_list = [current_board] * 4
        current_max_list = [current_max_element] * 4

        while not self._game_obj.is_end():
            current_board_list_len = len(current_board_list)
            np_current_board = np.array(current_board_list[current_board_list_len-4:])\
                .reshape(1, 4, 4, 4).astype(float)
            np_current_max = np.array(current_max_list[current_board_list_len-4:])\
                .reshape(1, 4).astype(float)
            action_qs = self._sess.run([self._actions_q],
                                       feed_dict={
                                           self._pl_board: np_current_board,
                                           self._pl_max: np_current_max,
                                           self._dropout_keep_prob: 1.0
                                       })[0][0]
            if random.random() > 0.01:
                actions, inc = self._game_obj.score_move(action_qs)
            else:
                actions, inc = self._game_obj.random_move()

            current_max_element = float(max(self._game_obj.get_board()))
            current_board = np.array(self._game_obj.get_board()) / current_max_element

            # float(max(current_board)
            current_board_list.append(current_board)
            current_max_list.append(current_max_element)

            np_next_board = np.array(current_board_list[len(current_board_list) - 4:])\
                .reshape(1, 4, 4, 4).astype(float)
            np_next_max = np.array(current_max_list[len(current_max_list) - 4:])\
                .reshape(1, 4).astype(float)
            next_action_qs = self._sess.run([self._actions_q],
                                            feed_dict={
                                                self._pl_board: np_next_board,
                                                self._pl_max: np_next_max,
                                                self._dropout_keep_prob: 1.0
                                            })[0][0]


            if self._game_obj.is_end():
                self._training_node_pool.append(
                    [np_current_board, np_current_max, actions, 0, self._game_obj.get_score()])
            else:
                self._training_node_pool.append(
                    [np_current_board, np_current_max, actions, max(next_action_qs), inc])
            if random.random() < 0.00001:
                print "For debug: " + str(self._training_node_pool[-1])

        self._game_step_pool.append(self._game_obj.get_action_counter())
        self._game_score_pool.append(self._game_obj.get_score())

        average_score = sum(self._game_score_pool) / len(self._game_score_pool)
        if self._game_obj.get_score() > average_score:
            self._game_beyond_pool.append(1)
        else:
            self._game_beyond_pool.append(0)

    def get_stat_info(self):
        """ Get internal stat info. """
        if len(self._game_score_pool) == 0:
            return 0.0, 0.0, 0.0, 0
        float_stat_pool_size = float(len(self._game_score_pool))
        average_score = sum(self._game_score_pool) / float_stat_pool_size
        avearge_step = sum(self._game_step_pool) / float_stat_pool_size
        avearge_beyond = sum(self._game_beyond_pool) / float_stat_pool_size
        self._hold_step = avearge_step / 2
        return len(self._training_node_pool), average_score, avearge_step, avearge_beyond, len(self._game_step_pool)

    def get_stat_info_string(self):
        stat_info=self.get_stat_info()
        return "pool_size:%d ave_score:%.1f ave_step:%.1f ave_beyond:%.3f stat_size:%d " % \
               (stat_info[0], stat_info[1], stat_info[2], stat_info[3], stat_info[4])

    def get_pool(self):
        return self._training_node_pool

    def create_external_feed_dict(self, pl_dict):
        """Creates the feed dict for the run process."""
        self.generate_training_data()
        self.generate_training_data()

        batch_nodes = random.sample(self._training_node_pool,
                                    self._batch_size)
        concat_nodes = self.__concat_nodes(batch_nodes)

        feed_dict = {
          pl_dict['c_b']: np.array(concat_nodes[0]).reshape(self._batch_size, 4, 4, 4).astype(float),
          pl_dict['c_m']: np.array(concat_nodes[1]).reshape(self._batch_size, 4).astype(float),
          pl_dict['c_a']: np.array(concat_nodes[2]).reshape(self._batch_size, 4).astype(float),
          pl_dict['c_r']: np.array(concat_nodes[3]).reshape(self._batch_size, 1).astype(float),
          pl_dict['dropout_keep_prob']: 1.0,
        }
        return feed_dict

    # Concatenated the nodes and return each concatenated fields separated.
    def __concat_nodes(self, batch_nodes):
        return map(lambda x: x[0], batch_nodes), \
               map(lambda x: x[1], batch_nodes), \
               map(lambda x: x[2], batch_nodes), \
               map(lambda x: x[3] + x[4], batch_nodes)

if __name__ == "__main__":
    import game_model
    import tensorflow as tf
    batch_size = 1
    with tf.Session() as sess:
        model = game_model.GameModel(batch_size)
        actions_q, conv1, conv2 = model.generate_action_q("train")
        init = tf.initialize_all_variables()
        sess.run(init)
        pool = GamePool(10000, sess, model.get_pl_dict(), actions_q, batch_size)
        pool.generate_training_data()

        print pool.create_external_feed_dict(model.get_pl_dict())

        for v in pool.get_pool():
            print v
