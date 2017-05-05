import tensorflow as tf
import numpy as np

with tf.name_scope('init') as scope:
    seed = np.random.seed()
    x = tf.random_uniform([5],minval=0.0,maxval=10.0,
                         dtype=tf.float32, seed=seed)
    x_sum = tf.summary.histogram('x', x)
with tf.name_scope('mean') as scope:
    mean = tf.reduce_mean(x)
    mean_sum = tf.summary.scalar('mean', mean)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/example', sess.graph)
    sess.run(init)
    for i in range(5):
        sess.run(x)
        sess.run(mean)
        summary = sess.run(merged)
        writer.add_summary(summary, i)
        print('X: ', sess.run(x),
              'MEAN: ', sess.run(mean))