import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')

sess = tf.Session()

print(hello)  # 텐서 출력(#print out operation)

print(sess.run(hello))

a = tf.constant(2)
b = tf.constant(3)

c = a*b

# print out operation
print(c)

# print out the result of operation
print(sess.run(c))


a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" %sess.run(a+b))
    print("Multiplication with constants; %i" %sess.run(a+b))


a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
    print("Addition with variables: %i"%sess.run(add, feed_dict={a:2, b:3}))
    print("Multiplication with variables: %i"%sess.run(mul, feed_dict={a:3, b:5}))


