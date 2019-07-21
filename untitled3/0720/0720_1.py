# MNIST data reconstruction with Auto Encoder (accuracy: 94%)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# mnist 데이터 다운로드
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# 학습에 필요한 설정값들을 정의
# 0717_4.py 와 동일
learning_rate = 0.02
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10
input_size = 784
hidden1_size = 256
hidden2_size = 128

# 입력값
# 주의! 오토인코더는 '비지도학습'이므로 타겟 레이블 y가 필요하지 않다.
x = tf.placeholder(tf.float32, shape=[None, input_size])


def build_auto_encoder(x):
    # 인코딩(Encoding) - 784 -> 256 -> 128
    w1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]), name='w1')
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]), name='b1')
    h1_output = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]), name='w2')
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]), name='b2')
    h2_output = tf.nn.sigmoid(tf.matmul(h1_output, w2) + b2)

    # 디코딩(Decoding) 128 -> 256 -> 784
    w3 = tf.Variable(tf.random_normal(shape=[hidden2_size, hidden1_size]), name='w3')
    b3 = tf.Variable(tf.random_normal(shape=[hidden1_size]), name='b3')
    h3_output = tf.nn.sigmoid(tf.matmul(h2_output, w3) + b3)

    w4 = tf.Variable(tf.random_normal(shape=[hidden1_size, input_size]), name='w4')
    b4 = tf.Variable(tf.random_normal(shape=[input_size]), name='b4')
    reconstructed_x = tf.nn.sigmoid(tf.matmul(h3_output, w4) + b4)

    return reconstructed_x


# 오토인코더를 선언
y_pred = build_auto_encoder(x)

# 타겟 데이터는 인풋 데이터와 같습니다.
y_true = x

# 손실 함수
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

# 옵티마이저
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_xs})

        if epoch % display_step == 0:
            print("반복(Epoch): %d, 손실 함수(loss): %f" % ((epoch+1), current_loss))

    reconstructed_result = sess.run(y_pred, feed_dict={x: mnist.test.images[: examples_to_show]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))

    f.savefig('reconstructed_mnist_image.png')
