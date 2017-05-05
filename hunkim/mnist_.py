import tensorflow as tf

x = tf.placeholder("float", [None, 784]) #mnist data image of shape 28*28 = 784
y = tf.placeholder("float", [None, 10]) # 0~9 digits recognition ==> 10 classes

#Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable((tf.zeros([10])))

#  Construct model
activation = tf.nn.sotfmax(tf.matmul(x, W)+b)

# Minimize error using cross entropy
cost = tf.reduce_maen(-tf.reduce_sum(y*tf.log(activation), reduction_indeices=1))
optimizer = tf.train.GradientEescentOptimizer(learning_rate).minimize(cost)

# Training cyckle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_vbatch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch)
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})/total_batch
    #Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:",'%04d'%(epoch+1), "cost=", "{:.9f".format(avg_cost))
print("Optimization Finished!")
#Test Model
correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

#Get one and predict
r = randint(0, mnist.test.num_examples -1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction: ", sess.run(Tf.argmax(activation,1), {x: mnist.test.images[r:r+1]}))

plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()

