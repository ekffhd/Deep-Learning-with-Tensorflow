# mnist basic classifier with ANN

import tensorflow as tf

# MNIST 데이터 다운로드
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 학습을 위한 설정 값 정의
learning_rate = 0.001
num_epochs = 30     # 학습 횟수
batch_size = 256    # 배치 개수
display_step = 1    # 손실함수 출력 주기
input_size = 784    # 입력 데이터 크기 28x28
hidden1_size = 256  # 은닉층1 크기
hidden2_size = 256  # 은닉층2 크기
output_size = 10    # output 크기

# 입력값과 출력값을 받기 위한 플레이스 홀더 정의
x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
y = tf.placeholder(tf.float32, shape=[None, output_size], name='y')


# ANN 모델 정의
def build_ann(x):
    w1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]), name='w1')
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]), name='b1')
    h1_output = tf.nn.relu(tf.matmul(x, w1)+b1)             # None x hidden1_size

    w2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]), name='w2')
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]), name='b2')
    h2_output = tf.nn.relu(tf.matmul(h1_output, w2) + b2)   # None x hidden2_size

    w_output = tf.Variable(tf.random_normal(shape=[hidden2_size, output_size]), name='w_output')
    b_output = tf.Variable(tf.random_normal(shape=[output_size]), name='b_output')
    logits = tf.matmul(h2_output, w_output) + b_output

    return logits


# ANN 모델 선언
predicted_value = build_ann(x)

# 손실 함수 계산
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=y))
tf.summary.scalar('loss', loss)

# 옵티마이저
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# 세션을 열고 그래프를 실행합니다.
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    tensorboard_writer = tf.summary.FileWriter('./0717_4_logs', sess.graph)

    for epoch in range(num_epochs):
        average_loss = 0.

        # 전체 배치를 불러온다.
        total_batch = int(mnist.train.num_examples/batch_size)

        x_train = []
        y_train = []
        # 모든 배치들에 대해서 최적화를 수행한다.
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})

            # 평균 손실함수 측정
            average_loss += current_loss / total_batch
            summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            tensorboard_writer.add_summary(summary,  i)

        if epoch % display_step == 0:
            print("반복(Epoch): %d, 손실함수(loss): %f" % ((epoch+1), average_loss))

        # 테스트 데이터를 이용해서 학습된 모델이 얼마나 정확한지 정확도를 출력
        correct_prediction = tf.equal(tf.argmax(predicted_value, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("정확도(Accuracy): %f" % (accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})))

