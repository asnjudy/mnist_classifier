import tensorflow as tf
from tf1x_common_utils import mnist_load_data
import numpy as np

# 获取数据
mnist = mnist_load_data()


def train_and_evaluate_helper(output, x, y_, keep_prob):
    # 交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=output))
    # 选择梯度优化算法
    train_step = tf.train.AdamOptimizer().minimize(loss)

    correction_predictions = tf.equal(tf.argmax(output, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correction_predictions, dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_size = 128
        num_steps = 1000
        eval_per_steps = 30

        for step in range(num_steps):
            # 获取一批训练数据
            train_batch = mnist.train.next_batch(batch_size)
            _, _train_loss = sess.run([train_step, loss],
                                      feed_dict={
                                          x: train_batch[0],
                                          y_: train_batch[1],
                                          keep_prob: 0.5
                                      })

            # 每隔100步计算一下模型的准确度
            if step % eval_per_steps == 0:
                _train_acc, _train_loss = sess.run([accuracy, loss],
                                                   feed_dict={
                                                       x: train_batch[0],
                                                       y_: train_batch[1],
                                                       keep_prob: 1.0
                                                   })
                # 获取一批测试数据，计算准确度
                test_batch = mnist.test.next_batch(300)
                _test_acc, _test_loss, _model_output = sess.run([accuracy, loss, output],
                                                                feed_dict={
                                                                    x: test_batch[0],
                                                                    y_: test_batch[1],
                                                                    keep_prob: 1.0
                                                                })

                print('step {} - aac: {:.4f}, loss: {:.4f}, test acc: {:.4f}, test loss: {:.4f}'.format(
                    step, _train_acc, _train_loss, _test_acc, _test_loss))

                # 打印前10个数据的预测情况
                print('        targets:', np.argmax(test_batch[1][:10], 1))
                print('   model output:', np.argmax(_model_output[:10], 1))
                # print('       targets:', train_batch[1][:4])
                # print('  model_output:', _model_output[:4])


def alexnet_train():
    from tf1x_impl1.func_build_alexnet import alexnet

    # 占位符
    x = tf.placeholder(tf.float32, shape=(None, 784), name='model_inputs')
    y_ = tf.placeholder(tf.int32, shape=(None, 10), name='model_targets')
    keep_prob = tf.placeholder(dtype=tf.float32, name='model_fc_dropout')

    # 前向计算获得模型输出
    output = alexnet(x, keep_prob)

    # 训练并评估模型
    train_and_evaluate_helper(output, x, y_, keep_prob)


def alexnet_train2():
    from tf1x_impl1.alexnet_model import AlexNetModel

    model = AlexNetModel()
    output = model.output  # 获取模型的前向计算结果

    x = model.x
    y_ = model.y_
    keep_prob = model.keep_prob
    # 训练并评估模型
    train_and_evaluate_helper(output, x, y_, keep_prob)


def alexnet_train3():
    from tf1x_impl1.alexnet_model import AlexNetModel2

    x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='model_inputs')
    y_ = tf.placeholder(dtype=tf.int32, shape=(None, 10), name='model_targets')
    # 定义全连接层 Dropout 概率（层中的每个神经元以 keep_prob 参与梯度更新）
    keep_prob = tf.placeholder(dtype=tf.float32, name='model_fc_dropout')

    model = AlexNetModel2()
    output = model(x, keep_prob=keep_prob)
    train_and_evaluate_helper(output, x, y_, keep_prob)
