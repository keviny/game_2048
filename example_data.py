import tensorflow as tf


def read_csv_format(filename_queue):
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)
  record_defaults = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1]]
  col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 = tf.decode_csv(
      value, record_defaults=record_defaults)
  feature = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10])
  label = col11
  return feature, label


def read_batch_data(filename_queue, batch_size, num_epochs=None):
  #filename_queue = tf.train.string_input_producer(
  #    file_name_list, num_epochs=num_epochs, shuffle=True)
  feature, label = read_csv_format(filename_queue)
  #min_after_dequeue = 10
  #capacity = min_after_dequeue + 3 * batch_size
  #example_batch, label_batch = tf.train.shuffle_batch(
  #    [feature, label], batch_size=batch_size, capacity=capacity,
  #    min_after_dequeue=min_after_dequeue)
  return feature, label  

if __name__=="__main__":
  tf.reset_default_graph()
  file_list = "a0.csv"
  filename_map = {'a0.csv':0}
  filename_queue = tf.FIFOQueue(capacity=4, dtypes=[tf.string])
  #config = tf.ConfigProto(log_device_placement=True)
  #config.set_operation_timeout_in_ms(1500)   # terminate on long hangs
  init_op = tf.global_variables_initializer()
  config = tf.ConfigProto(inter_op_parallelism_threads=2)
  with tf.Session(config=config) as sess:
    sess.run(init_op)
    enqueue_op = filename_queue.enqueue(file_list)
    sess.run(enqueue_op)
    print 'size before', sess.run(filename_queue.size())
    features, labels = read_batch_data(filename_queue, 3)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    counter = 0
    print "start loop"
    try:
      while not coord.should_stop():
        counter = counter + 1
        print "a"
        value = sess.run([features])
        print value
        if counter % 10 == 0:
          print counter, sum(value)
          print dir(filename_queue)
          index = (counter / 10) % 3
          filename = 'a%d.csv' % index
          if filename not in filename_map.keys():
            filename_map[filename] = 0
            sess.run(filename_queue.enqueue("a%d.csv" % index))
          print 'size %d' % sess.run(filename_queue.size())
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      print "final"
      coord.request_stop()
  print "aaaaa"
  coord.join(threads)
  print "end"
