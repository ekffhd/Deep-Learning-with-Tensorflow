# MNIST data classification with Auto Encoder + Softmax (accuracy: 96%)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# mnist 데이터 다운로드
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# 학습에 필요한 설정값들을 정의
learning_rate_RMSProp = 0.02
learning_rate_GradientDescent = 0.5
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10
input_size = 784
hidden1_size = 256
hidden2_size = 64

# 입력값
x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y')


def build_auto_encoder(x):
    # 인코딩(Encoding) - 784 -> 256 -> 64
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

    return reconstructed_x, h2_output


# Softmax 분류기 정의
def build_softmax_classifier(x):

    # MNIST 원본 데이터 대신 오토인코더의 압축된 특징값(64)을 입력값으로 받는다.
    w_softmax = tf.Variable(tf.zeros([hidden2_size, 10]))
    b_softmax = tf.Variable(tf.zeros([10]))
    y_pred = tf.nn.softmax(tf.matmul(x, w_softmax) + b_softmax)

    return y_pred


# 오토인코더를 선언
y_pred, extracted_features = build_auto_encoder(x)

# 타겟 데이터는 인풋 데이터와 같다.
y_true = x

# Softmax분류기 선언
y_pred_softmax = build_softmax_classifier(extracted_features)

# 1. Pre-Training :MNIST 데이터 재구축을 목적으로하는 손실 함수와 옵티마이저 정의
pretraining_loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
pretraining_train_step = tf.train.RMSPropOptimizer(learning_rate_RMSProp).minimize(pretraining_loss)
tf.summary.scalar('pre-training loss', pretraining_loss)

# 2. Fine-Tuning : MNIST 데이터 분류를 목적으로 하는 손실 함수와 옵티마이저 정의
# cross-entropy loss 함수
finetuning_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred_softmax), reduction_indices=[1]))
finetuning_train_step = tf.train.GradientDescentOptimizer(learning_rate_GradientDescent).minimize(finetuning_loss)
tf.summary.scalar('fine-tunining loss', finetuning_loss)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    tensorboard_writer = tf.summary.FileWriter('./0721_1_logs', sess.graph)

    total_batch = int(mnist.train.num_examples / batch_size)

    # Step1 : MNIST 데이터 재구축을 위한 오토인코더 최적화
    for epoch in range(training_epochs):

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, pretraining_loss_print = sess.run([pretraining_train_step, pretraining_loss], feed_dict={x: batch_xs})

        if epoch % display_step == 0:
            print("반복(Epoch): %d, Pre-training 손실 함수(pretraining_loss): %f" % ((epoch+1), pretraining_loss_print))

        pretraining_summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})
        tensorboard_writer.add_summary(pretraining_summary, i)

    # Step2 : MNIST데이터 분류를 위한 오토인코더 + Softmax 분류기 최적화
    for epoch in range(training_epochs+10):

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, finetuning_loss_print = sess.run([finetuning_train_step, finetuning_loss],
                                                feed_dict={x: batch_xs, y: batch_ys})

        if epoch % display_step == 0:
            print("반복(Epoch): %d, Fine-tuning 손실 함수(finetuning_loss): %f" % ((epoch + 1), finetuning_loss_print))

        finetuning_summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys})
        tensorboard_writer.add_summary(finetuning_summary, i)

    print("Step 2: MNIST 데이터 분류를 위한 오토인코더 + Softmax 분류기 최적화 완료(Fine-Tuning)")

    reconstructed_result = sess.run(y_pred, feed_dict={x: mnist.test.images[: examples_to_show]})

    # 오토인코더 + softmax 분류기 모델의 정확도를 출력한다.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred_softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("정확도(오토인코더 + Softmax 분류기 %f)" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
