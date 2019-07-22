# Classify CIFAR-10 images with CNN (accuracy: 0.68)
import tensorflow as tf
import numpy as np

"""
tf.nn.dropout(x, keep_prob, name = None)

x           : 드롭아웃을 적용할 인풋 데이터
keep_prob   : 드롭하지 않고 유지할 비율을 나타내는 scalar 텐서
name        : 연산의 이름
"""
"""
tf.nn.max_pool

padding     : 나누어 떨어지지 않는 경우에 패딩을 추가한다.
"""
# CIFAR-10 데이터 다운로드
from tensorflow.keras.datasets.cifar10 import load_data


# 'num' 개수 만큼 랜덤한 샘플들과 레이블들을 리턴
def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# CNN 모델 정의 : 5개의 Convolution Layer , 2개의 Pooling, 2개의  Full-connected-Layer
def build_cnn_classifier(x):
    # input image : 32x32x3
    x_image = x

    # 1st Convolution Layer
    # 하나의 RGB 이미지를 64개의 특징들로 맵핑
    # 32x32x3 -> 32x32x64
    w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2), name='w_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv1')
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, name='h_conv1')

    # 1st Pooling
    # 32x32x64 -> 16x16x64
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 2nd Convolution Layer
    # 64개의 특징들을 64개의 특징들로 맵핑
    # 16x16x64 -> 16x16x64
    w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2), name='w_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, name='h_conv2')

    # 2nd Pooling
    # 16x16x64-> 8x8x64
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')

    # 3rd Convolution Layer
    # 64개의 특징들을 128개의 특징들로 맵핑
    # 8x8x64 -> 8x8x128
    w_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2), name='w_conv3')
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv3')
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3, name='h_conv3')

    # 4th Convolution Layer
    # 128개의 특징들을 128개의 특징들로 맵핑
    # 8x8x128 -> 8x8x128
    w_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2), name='w_conv4')
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv4')
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4, name='h_conv4')

    # 5th Convolution Layer
    # 128개의 특징들을 128개의 특징들로 맵핑
    # 8x8x128 -> 8x8x128
    w_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2), name='w_conv5')
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]), name='b_conv5')
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, w_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5, name='h_conv5')

    # 1st Full Connected Layer
    # 8x8x128 -> 384
    w_fc1 = tf.Variable(tf.truncated_normal(shape=[8*8*128, 384], stddev=5e-2), name='w_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]), name='b_fc1')

    h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, w_fc1) + b_fc1)

    # Drop-Out
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 2nd Full Connected Layer
    # 384 -> 10
    w_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2), name='w_fc2')
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc2')
    logits = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits


# 인풋, 아웃풋 데이터와 드롭아웃 확률을 받기위한 플레이스홀더 정의
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# CIFAR-10 데이터 다운로드 및 로드
(x_train, y_train), (x_test, y_test) = load_data()

# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환
"""
tf.squeeze([[0],[1],[2,]])
-> array([0, 1, 2])
차수 줄인다.
"""
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

# build_cnn_classifier 함수를 이용하여 CNN 그래프를 선언
# 크로스 엔트로피 손실함수와 0.001의 런닝레이트를 가진 RMSPropr 옵티마이저 선언
y_pred, logits = build_cnn_classifier(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
tf.summary.scalar('loss', loss)

# 정확도 출력 연산
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    tensorboard_writer = tf.summary.FileWriter('./tmp_2/logs', sess.graph)

    # 1000 step 만큼 최적화
    for i in range(10000):

        batch = next_batch(128, x_train, y_train_one_hot.eval())

        # 10 Step 마다 training 데이터셋에 대한 정확도와 loss 출력
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

            print("반복(Epoch): %d, 트레이딩 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))

        # 20% 확률의 드롭아웃을 이용해서 학습을 진행
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})

        summary = sess.run(merged, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})
        tensorboard_writer.add_summary(summary, i)

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})

    test_accuracy = test_accuracy / 10
    print("테스트 데이터 정확도: %f" % test_accuracy)

