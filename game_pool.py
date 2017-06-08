import time
import game
import random
import numpy as np
from collections import deque


class GamePool(object):
    MODEL_GENERATOR = 1
    RANDOM_GENERATOR = 2

    def __init__(self, pool_size, sess, model):
        """
        The training data pool is a collection of each training record.
        Each training record is composed of (s0, a0, r0, s1)
        s0, s1 represents the current board status and next step board status.
        a0 represents the action executed in s0
        r0 represents the incremental score of this execution

        Args:
          pool_size: The number of record this pool could contain
          sess: Session variable.
          model: The model object.
        """
        self._sess = sess
        self._batch_size = model.get_batch_size()
        self._actions_q = model.get_internal_variable().get("current_q")
        self._model = model
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

        while not self._game_obj.is_end():
            current_board = np.array(self._game_obj.get_before_board())

            feed_dict = self._model.create_feed_dict(4, self._game_obj.get_eval_boards(), None, None, None)
            # self._sess.run([self._actions_q], feed_dict=feed_dict)
            action_qs = self._sess.run([self._actions_q], feed_dict=feed_dict)[0]
            action_score = action_qs + self._game_obj.get_eval_inc()
            #print action_qs,self._game_obj.get_eval_inc(), action_score
            #print
            # use random move for some case.
            if random.random() > 0.1:
                actions, inc, before_gen_board = self._game_obj.score_move(action_score)
            else:
                actions, inc, before_gen_board = self._game_obj.random_move()

            if self._game_obj.is_end():
                pass
                # self._training_node_pool.append([current_board, self._game_obj.get_before_board(), actions, self._game_obj.get_score(), self._game_obj.get_score()])
            else:
                self._training_node_pool.append(
                    [current_board, self._game_obj.get_before_board(), actions, self._game_obj.get_score(), inc])
                # print self._training_node_pool[-1]
            if random.random() < 0.00001:
                print "For debug: " + str(self._training_node_pool[-1])

        # For statsself._training_node_pool
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

    def create_external_feed_dict(self, game_model_instance):
        """Creates the feed dict for the run process."""
        batch_nodes = random.sample(self._training_node_pool, game_model_instance.get_batch_size())
        concat_nodes = self.__concat_nodes(batch_nodes)

        feed_dict = game_model_instance.create_feed_dict(
            game_model_instance.get_batch_size(),
            concat_nodes[0],
            concat_nodes[1],
            concat_nodes[2],
            concat_nodes[3])
        return feed_dict

    # Concatenated the nodes and return each concatenated fields separated.
    def __concat_nodes(self, batch_nodes):
        return map(lambda x: x[0], batch_nodes), \
               map(lambda x: x[1], batch_nodes), \
               map(lambda x: x[2], batch_nodes), \
               map(lambda x: x[4], batch_nodes)

if __name__ == "__main__":
    import game_model
    import tensorflow as tf
    batch_size = 1
    with tf.Session() as sess:
        model = game_model.GameModel(batch_size)
        actions_q, _, _ = model.generate_action_q("train")
        init = tf.global_variables_initializer()
        sess.run(init)
        pool = GamePool(10000, sess, model.get_pl_dict(), actions_q, batch_size)
        pool.generate_training_data()

        print pool.create_external_feed_dict(model.get_pl_dict())

        for v in pool.get_pool():
            print v
