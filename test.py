import tensorflow as tf
import time
import numpy as np

class game(object):
    def __init__(self):
        self.screen_size = [1,1]
        self.counter = 0

    def grab_screen(self):
        """current screen of the game"""
        self.counter += 1
        screen = self.counter*np.ones(self.screen_size)
        return screen

experience = tf.RandomShuffleQueue(10000,
                                10, tf.float32,
                                shapes = [1,1],
                                name = 'experience_replay')

def perceive(game):
    rawstate = game.grab_screen()
    enq = experience.enqueue(rawstate)
    return enq

available_threads = 1

game_list = [game() for x in xrange(available_threads)]
#create threads to play the game and collect experience

coord = tf.train.Coordinator()
experience_runner = tf.train.QueueRunner(
  experience, [perceive(game_list[num]) for num in range(available_threads)])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
enqueue_threads = experience_runner.create_threads(sess, coord = coord, start = True)

with sess.as_default():
    while(1):
        print sess.run(experience.dequeue())
