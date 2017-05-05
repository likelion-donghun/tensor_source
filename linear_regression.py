import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

point_num = 1000
x_data = [0]*point_num
y_data = [0]*point_num

for i in range(point_num):
    x_data[i]=np.random.normal(0.0, 0.5)
    y_data[i]=x_data[i]+np.random.normal(0.0, 0.3)

x_data[i] = np.random.normal(0.0, 0.5)
y_data[i] = x_data[i]+np.random.normal(0.0, 0.3)

plt.plot(x_data, y_data, 'ro')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

with tf.name_scope("init") as scope:
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0),name='W')
    b = tf.Variable(tf.zeros([1]),name='b')
    y = W * x_data + b

with tf.name_scope("loss") as scope:
    learning_rate = 0.05
    cost = tf.reduce_mean(tf.square(y - y_data))
    cost_sum = tf.summary.histogram('loss',cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

def plt_print():
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/linear", sess.graph)
    for step in range(2000):
        sess.run(train)
        if step%200 == 0:
            print(step, sess.run(W), sess.run(b))
            print(step, sess.run(cost))
        summary = sess.run(merged)
        writer.add_summary(summary,step)
    plt_print()




