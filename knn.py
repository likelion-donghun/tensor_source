import tensorflow as tf
import numpy as np
import time

start = time.time()

with tf.name_scope('init')as scope:
    with tf.name_scope('example')as scope:
        x = np.loadtxt('a.csv', delimiter=',', unpack=True, dtype=np.float32)
        x_label = x[0][0]
        x_data = np.transpose(x[1:])

        y = np.loadtxt('b.csv', delimiter=',', unpack=True, dtype=np.float32)
        y_label = y[0][0]
        y_data = np.transpose(y[1:])
    with tf.name_scope('new')as scope:
        input1 = np.loadtxt('input1.csv', delimiter=',', dtype=np.float32)
        input1_label = input1[0]
        input1_data = input1[1:]

        input2 = np.loadtxt('input2.csv', delimiter=',', dtype=np.float32)
        input2_label = input2[0]
        input2_data = input2[1:]
with tf.name_scope('cal_dist')as scope:
    data = []
    feed_data = tf.placeholder('float', [2000, 2])


    def get_dist(a, input_data):
        distance = tf.reduce_sum(tf.square(tf.subtract(a, input_data)))
        return distance

with tf.name_scope('update') as scope:
    a_count = tf.Variable(0)
    b_count = tf.Variable(0)
    ac_update = tf.assign_add(a_count, 1)
    bc_update = tf.assign_add(b_count, 1)
    check = tf.less(a_count, b_count)
k = 11
value, indices = tf.nn.top_k(feed_data[:, 1], k=k, sorted=True)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # writer = tf.summary.FileWriter('./logs/k-nn',sess.graph)

    print('x_dist\n')
    for step in range(len(x_data)):
        dist = -1 * sess.run(get_dist(x_data[step], input2_data))
        data.append([x_label, dist])

    print('y_dist\n')
    for step in range(len(y_data)):
        dist = -1 * sess.run(get_dist(y_data[step], input2_data))
        data.append([y_label, dist])

    print('sort\n')
    dist = sess.run(value, feed_dict={feed_data: data})
    dist_r = tf.multiply(dist, -1)
    indices_r = sess.run(indices, feed_dict={feed_data: data})
    print(sess.run(dist_r))
    print(indices_r)

    for i in range(k):
        idx = sess.run(indices[i], feed_dict={feed_data: data})
        if sess.run(tf.less(idx, 1000)):
            sess.run(ac_update)
        else:
            sess.run(bc_update)
    print('a:', sess.run(a_count), '\nb:', sess.run(b_count))
    if sess.run(check) != True:
        print("1 입니다.")
    else:
        print("2 입니다.")
    print("끝")

    end = time.time()
    print("총시간 : ", (end - start)/60)