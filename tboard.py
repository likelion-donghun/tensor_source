import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

with tf.name_scope('image') as scope:
    filename = 'cat.jpg'
    image = mpimg.imread(filename)
    height, width, color = image.shape
    image_shaped_input = tf.reshape(image,[-1,height, width,3])
    image_sum = tf.summary.image('input', image_shaped_input,3)
    print(image.shape)
    x = tf.Variable(image, name='x')
    transpose_x = tf.transpose(x, perm = [1, 0, 2], name='transpose')
    transpose_img = tf.reshape(transpose_x, [-1, width,height,3])
    output_sum = tf.summary.image('output', transpose_img, 3)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs/Do_plt', sess.graph)
    summary = sess.run(merged)
    writer.add_summary(summary)
    result = sess.run(transpose_x)

plt.imshow(result)
plt.show()