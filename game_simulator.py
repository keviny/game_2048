import argparse
import game
import logging
import numpy as np
import game_model
import random
import time
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
  '-d', '--debug',
  help="Print lots of debugging statements",
  action="store_const", dest="loglevel", const=logging.INFO,
  default=logging.INFO)

parser.add_argument('--use_ai', help="choose to use ai", action="store_true", default=False)

parser.add_argument('--model_path')

parser.add_argument('--log_name', default='game.log')

parser.add_argument('--exp_name', help="choose to use ai", default='d1')

args = parser.parse_args()

logging.basicConfig(filename=args.log_name, level=args.loglevel)


class GameSimulator(object):
  def __init__(self, game, model_path, use_ai=True, exp_name="d1"):
    self.game = game
    self.model = game_model.GameModel("train", 1)
    self.target_board_q = self.model.get_internal_variable().get("action_q")
    self.sess = tf.Session()
    self._use_ai = use_ai

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path + exp_name)
    saver.restore(self.sess, ckpt.model_checkpoint_path)

  def eval(self):
    board_q_value = self.sess.run(
          [self.target_board_q],
          feed_dict={
            self.model.get_pl_dict().get("c_b"): np.array(self.game.get_board()).reshape(1, 4, 4, 1).astype(float),
            self.model.get_pl_dict().get("c_m"): np.array(max(self.game.get_board())).reshape(1, 1).astype(float),
            self.model.get_pl_dict().get("dropout_keep_prob"): 1.0
          })
    return board_q_value[0][0]

  def get_game(self):
    return self.game;

  def move(self):
    raw_directions = self.game.get_valid_directions(False)
    board_score_dict = self.game.get_next_boards(raw_directions)
    if self._use_ai:
      next_score_list = self.eval()
    else:
      next_score_list = map(lambda k: board_score_dict[k][1], board_score_dict)[0:4]
    # max score must be in the set of valid direction.
    max_score = max(map(lambda i: next_score_list[i], self.game.get_valid_directions()))
    max_direction = []
    for i in xrange(0, 4):
      # The direction must be valid direction.
      if i in self.game.get_valid_directions() and abs(next_score_list[i] - max_score) < 0.0001:
        max_direction.append(i)
    self.game.move(random.choice(max_direction))


if __name__ == "__main__":
  simulator = GameSimulator(game.Game(), args.model_path, args.use_ai, args.exp_name)
  step_list = []    # For stat
  score_list = []   # For stat
  total_simulation_counter = 1000
  for x in xrange(total_simulation_counter):
    simulator.get_game().reset()
    while len(simulator.get_game().get_valid_directions()) > 0:
      simulator.move()
    simulator.get_game().display()
    log_content = "step=%d\tmax=%d\tscore=%d" % (simulator.get_game().get_action_counter(),
                                  max(simulator.get_game().get_board()),
                                  simulator.get_game().get_score())
    logging.info("Record id:%05d info: %s" % (x, log_content))
    print "Final result: %05d %s" % (x, log_content)
    step_list.append(simulator.get_game().get_action_counter())
    score_list.append(simulator.get_game().get_score())
  print "Average step %d, Average score %d" % (sum(step_list) / total_simulation_counter,
                                               sum(score_list) / total_simulation_counter)
