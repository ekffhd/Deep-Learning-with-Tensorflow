# Tensorflow basic

import tensorflow as tf

W = tf.Variable(tf.random_normal(shape=[1]))
b = tf.Variable(tf.random_normal(shape=[1]))
x = tf.placeholder(tf.float32)
linear_model = W*x + b

y = tf.placeholder(tf.float32)

loss = tf.reduce_mean(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step, feed_dict={x: x_train, y: y_train})

x_test = [3, 3.5, 5, 6]
print(sess.run(linear_model, feed_dict={x: x_test}))

sess.close()
