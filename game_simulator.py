import argparse
import game
import logging
import numpy as np
import game_model
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--debug',
    help="Print lots of debugging statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.INFO)

parser.add_argument('--use_ai', help="choose to use ai", action="store_true", default=False)

parser.add_argument('--model_path')

parser.add_argument('--log_name', default='game.log')

parser.add_argument('--exp_name', help="choose to use ai", default='d1')


args = parser.parse_args()

logging.basicConfig(filename=args.log_name, level=args.loglevel)


class GameSimulator(object):
  def __init__(self, model_path, use_ai=True, exp_name="d1"):
    self.model = game_model.GameModel("train", 1)
    self.target_board_q = self.model.get_internal_variable().get("action_q")
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    print self.sess.run(self.model.get_params_dict().get("model_conv1_weights"))
    self.saver = tf.train.import_meta_graph(model_path + "%s/model.meta" % exp_name)
    self.saver.restore(self.sess, model_path + "%s/model.ckpt" % exp_name)
    self._use_ai = use_ai
    print self.sess.run(self.model.get_params_dict().get("model_conv1_weights"))

  def eval(self, eval_board):
      """
      This function calcuates the possbile value of next step and
      and returns max value of the sum of q value and incrmental score.
      """
      internal_game = game.Game()
      internal_game.load_board(eval_board)
      internal_game.random_gen()
      raw_directions = internal_game.get_valid_directions(False)
      board_score_dict = internal_game.get_next_boards(raw_directions)
      valid_directions = filter(lambda x: x != -1, raw_directions)
      next_total_score = []
      for direction in [game.Direction.left,
                        game.Direction.right,
                        game.Direction.up,
                        game.Direction.down]:
          if direction not in valid_directions:
              # If the direction is not valid, add a negative value in it.
              next_total_score.append(-sys.maxint)
              continue
          move_board, move_score = board_score_dict[direction]
          max_board = max(move_board)
          if self._use_ai == False:
            next_total_score.append(move_score)
          else:
            board_q_value = self.sess.run(
                [self.target_board_q],
                feed_dict={
                  self.model.get_pl_dict().get("c_b"): np.array(move_board).reshape(1, 4, 4, 1).astype(float),
                  self.model.get_pl_dict().get("c_m"): np.array(max_board).reshape(1, 1).astype(float),
                  #self.model.get_pl_dict().get("c_r"): np.array([1.0]).reshape(1, 1).astype(float),
                  #self.model.get_pl_dict().get("c_a"): np.array([1.0, 1.0, 1.0, 1.0]).reshape(1, 4).astype(float),
                  self.model.get_pl_dict().get("dropout_keep_prob"): 1.0
                  })
            # Adds the q value of next step board and incrmental score
            next_total_score.append(board_q_value[0][0][0] + move_score)
      return max(next_total_score), next_total_score

  def move(self, current_board):
      internal_game = game.Game()
      internal_game.load_board(current_board)
      raw_directions = internal_game.get_valid_directions(False)
      board_score_dict = internal_game.get_next_boards(raw_directions)
      logging.debug(board_score_dict)
      valid_directions = filter(lambda x: x != -1, raw_directions)
      # The score array for the first layer of movement.
      next_total_score = []
      # The score array fro the second layer of movement.
      next_move_score = []
      for direction in [game.Direction.left,
                        game.Direction.right,
                        game.Direction.up,
                        game.Direction.down]:
          logging.debug(direction)
          if direction not in valid_directions:
              # If the direction is not valid, add a negative value in it.
              next_total_score.append(-sys.maxint)
              continue
          move_board, move_score = board_score_dict[direction]
          next_best_score, next_score_array = self.eval(move_board)
          next_move_score.append({direction:next_score_array})
          next_total_score.append(next_best_score + move_score)
      target_move = np.argmax(np.array(next_total_score))
      return target_move, next_total_score, next_move_score

if __name__ == "__main__":
  simulator = GameSimulator(args.model_path, args.use_ai, args.exp_name)
  for x in xrange(100):
    target_game = game.Game()
    while len(target_game.get_valid_directions()) > 0:
      if args.loglevel==logging.DEBUG:
        target_game.display()
      move_direction, first_layer_score, second_layer_score = simulator.move(target_game.get_board())
      target_game.move(move_direction)
      logging.debug("%s %s" % (first_layer_score, second_layer_score))
    log_content = "%d:%f" % (target_game.get_action_counter(), target_game.get_score())
    logging.info("Final result: %05d %s" % (x, log_content))
    print "Final result: %05d %s" % (x, log_content)
    target_game.display()