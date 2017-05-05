import tensorflow as tf
#Minimize error using cross entropy
learning_rate = 10.0

#cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

#Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y: y_data}), sess.run(W))