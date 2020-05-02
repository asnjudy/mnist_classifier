import tensorflow as tf
from models.utils import create_weight_variable, create_bias_variable, conv_op, fc_op


class AlexNetModel(object):
    def __init__(self):
        ''' 模型的参数，可以按需开放出来
        '''
        # 定义输入输出
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='model_inputs')
        self.y_ = tf.placeholder(dtype=tf.int32, shape=(None, 10), name='model_targets')

        # 3个卷积层
        self.conv_filter_shape_list = [(3, 3, 1, 64), (3, 3, 64, 128), (3, 3, 128, 256)]

        # 2个全连接层
        # 最后一层为最终的分类概率层，使用 softmax 激活
        # 前面的层使用 relu 激活函数
        self.fc_unit_num_list = [1024, 10]

        # 全连接层后接 Dropout
        # 设定全连接层神经元的保留概率
        # 所谓的“丢弃”，表示节点之间的连接权值在本次训练时不更新
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='model_fc_dropout')

        # 搭建模型
        self._build()

    def _build(self):
        ''' 搭建模型
        输入数据形状为：
            (None, 784) -> (None, 28, 28, 1)
            一批 28x28x1 的图片（像素值压缩到 0~1 之间）
        输入数据形状为：
            (None, 10), 为属于数字0~9这10个数的概率
        '''

        x_images = tf.reshape(self.x, [-1, 28, 28, 1])
        # 拼接卷积层
        conv_in = x_images
        for i, conv_filter_shape in enumerate(self.conv_filter_shape_list):
            conv_op_name = 'conv%s' % (i + 1)
            conv_out = conv_op(conv_in, filters_shape=conv_filter_shape, variable_scope_name=conv_op_name)
            print('##', conv_op_name, conv_out)  # 打印卷积形状
            conv_in = conv_out  # 当前卷积的输出作为下层卷积的输入

        # 拼接全连接层, conv_out 形状为(?, 4, 4, 256)
        # (None, d1, d2, d3) -> (None, d1*d2*d3)
        input_unit_num = (conv_out.shape[1] * conv_out.shape[2] * conv_out.shape[3]).value
        fc_in = tf.reshape(conv_out, shape=(-1, input_unit_num))
        for i, fc_unit_num in enumerate(self.fc_unit_num_list):
            fc_op_name = 'fc%s' % (i + 1)
            if i == len(self.fc_unit_num_list) - 1:
                # 最后的分类层，在损失函数中使用softmax交叉熵损失，无需激活
                fc_out = fc_op(fc_in, input_unit_num, fc_unit_num, fc_op_name)
            else:
                fc_out = fc_op(fc_in, input_unit_num, fc_unit_num, fc_op_name)
                fc_out = tf.nn.relu(fc_out)  # 激活
                fc_out = tf.nn.dropout(fc_out, self.keep_prob)  # Dropout

                # 当前层的输出作为下一层的输入
                input_unit_num = fc_unit_num
                fc_in = fc_out
            print('##', fc_op_name, fc_out)

        # 模型前向计算的输出
        self.output = fc_out

        # 训练模式，定义损失函数
        # 待学习的任务类型（多分类）决定了有哪些损失函数可以
        # 然后根据具体的模型选择一个合适的损失函数
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.output))

        # 评估指标，定义准确率
        # 模型准确度计算方法
        # self.y_ 为 targets, 为样例的实际标签
        # self.output, 为模型的最后一个全连接层（分类层）的输出。
        # 注意，此时还没有进行 softmax 归一化为概率，可以看成是打分
        # 选择最高打分值对应的索引作为分类的类别
        correction_predictions = tf.equal(tf.argmax(self.y_, axis=1), tf.argmax(self.output, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correction_predictions, dtype=tf.float32))

    def get_output(self):
        if self.output is None:
            raise Exception('please build the model')
        return self.output

    def get_loss(self):
        if self.loss is None:
            raise Exception('please build the model')
        return self.loss

    def get_accuracy(self):
        if self.accuracy is None:
            raise Exception('please build the model')
        return self.accuracy


def alexnet_train_with_session():
    def mnist_load_data():
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('./data', one_hot=True)
        return mnist

    def create_feed_dict(model, batch, keep_prob):
        return {model.x: batch[0], model.y_: batch[1], model.keep_prob: keep_prob}

    model = AlexNetModel()
    model_output = model.get_output()
    model_loss = model.get_loss()
    model_accuracy = model.get_accuracy()

    # 获取数据
    mnist = mnist_load_data()

    # 尝试跑一下模型
    # 前向计算，计算模型的输出
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch = mnist.train.next_batch(2)
        _output = sess.run(model_output, feed_dict=create_feed_dict(model, batch, 1))
        print(_output)

    # 选择梯度优化算法
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=model_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(1000):
            train_batch = mnist.train.next_batch(128)
            _, _train_loss = sess.run([train_step, model_loss],
                                      feed_dict=create_feed_dict(model, train_batch, 0.5))

            print('step', step, ', train loss:', _train_loss)
            # 每隔100步计算一下模型的准确度
            if step % 100 == 0:
                _train_acc, _train_loss = sess.run([model_accuracy, model_loss],
                                                   feed_dict=create_feed_dict(model, train_batch, 1.0))

                test_batch = mnist.test.next_batch(200)
                _test_acc, _test_loss = sess.run([model_accuracy, model_loss],
                                                 feed_dict=create_feed_dict(model, test_batch, 1.0))
                print('aac: {:.4f}, loss: {:.4f}, test acc: {:.4f}, test loss: {:.4f}'.format(
                    _train_acc, _train_loss, _test_acc, _test_loss))
