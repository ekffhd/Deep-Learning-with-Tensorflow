# Classify mnist data with CNN + softmax (accuracy: 98%)
import tensorflow as tf

# MNIST 데이터 다운로드
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp_1_data/", one_hot=True)


# CNN 모델 정의
def build_cnn_classifier(x):

    # 28x28 크기, grayscale 이미지 -> 컬러 채널 1
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 1st convolution layer
    # 5x5 Kernel Size를 가진 32개의 Filter 적용
    # 28x28x1 -> 28x28x32

    """
    * tf.truncated_normal
    
    절단 정규분포로부터의 난수값을 반환하는 함수
    stddev : 절단정규분포의 표준편차
    """

    """
    * tf.nn.conv2d(input, filter, strides, padding, dilation = [1, 1, 1, 1], name = None
    
    컨볼루션 연산을 구현하는 API
    input :     컨볼루션 연산을 적용할 인풋 데이터. [batch, in_height, in_width, in_channels]
                ex) 64개의 배치로 묶은 28x28 크기의 흑백 이미지 [64, 28, 28, 1]
                
    filter :    컨볼루션 연산에 적용할 필터. [filter_height, filter_width, in_channels, out_channels]
                - in_channels  : 인풋 데이터의 차원에 의해 결정
                - out_channels : 현재 컨볼루션 연산에서 사용할 필터 개수
    
    stride :    input 데이터의 배치, 채널은 1씩 건너뛰고, 가로, 세로 길이를 각각 2픽셀씩 건너뛴는 stride [1, 2, 2, 1]
    
    padding :   - SAME  : input 의 가로, 세로 차원과 똑같은 크기의 가로, 세로 차원이 리턴되도록 제로패딩을 적용
                - VALID : 컨볼루션 연산 공식에 의해 계산된 가로, 세로 차원 리턴
    """
    w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=5e-2), name='w_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, name='h_conv1')

    # 1st pooling
    # Max Pooling 을 이용해서 이미지의 크기를 1/2로 down-sampling
    # 28x28x32 -> 14x14x32
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')

    # 2nd convolution layer
    # 5x5 Kernel Size를 가진 64개의 Filter를 적용
    # 14x14x32 -> 14x14x64
    w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2), name='w_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv2')
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2, name='h_conv2')

    # 2nd pooling
    # Max Pooling 을 이용하여 이미지 크기를 1/2로 down-sampling
    # 14x14x64 -> 7x7x64
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')

    # 완전 연결층
    # 7x7 크기를 가진 64개의 activation map 을 1024개의 특징들로 변환
    # 7x7x64(3136) -> 1024
    w_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=5e-2), name='w_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')
    h_pool2_flat = tf.reshape(h_pool2, shape=[-1, 7*7*64], name='h_pool2_flat')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name='h_fc1')

    # 출력층
    # 1024개의 특징들을 10개의 클래스로 변환 (One-hot Encoding 으로 표현된 숫자)
    # 1024 -> 10
    w_output = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=5e-2), name='w_output')
    b_output = tf.Variable(tf.constant(0., shape=[10]))
    logits = tf.matmul(h_fc1, w_output) + b_output
    y_pred = tf.nn.softmax(logits, name='soft-max')

    return y_pred, logits


# 인풋과 아웃풋 데이터를 받을 플레이스 홀더를 정의
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

# CNN 분류기 선언
y_pred, logits = build_cnn_classifier(x)

# 손실함수와 옵티마이저 정의
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits), name='loss')
train_step = tf.train.AdamOptimizer(1e-4, name='Adam').minimize(loss)
tf.summary.scalar('loss', loss)

# 정확도 출력 연산
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    tensorboard_writer = tf.summary.FileWriter('./tmp_1/logs', sess.graph)

    # 2000step 최적화 수행
    # 10000step 최적화 할 경우 accuracy : 99%
    for i in range(2000):

        # 50개씩 MNIST 데이터 load
        batch = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
            print("반복(Epoch): %d, 트레이딩 데이터 정확도: %f" % (i, train_accuracy))

        sess.run([train_step], feed_dict={x: batch[0], y: batch[1]})
        summary = sess.run(merged, feed_dict={x: batch[0], y: batch[1]})
        tensorboard_writer.add_summary(summary, i)

    print("테스트 데이터 정확도 %f" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))

