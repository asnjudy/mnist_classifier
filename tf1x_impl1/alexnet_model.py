import tensorflow as tf
from tf1x_common_utils import *


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


class AlexNetModel2(object):
    def __init__(self):

        self.conv_filter_shape_list = [(3, 3, 1, 64), (3, 3, 64, 128), (3, 3, 128, 256)]
        self.fc_unit_num_list = [1024, 10]

    def __call__(self, x, keep_prob=1.0):
        """
        x 是待输入数据的占位符，是一个 tensor, 在 tf.Session 中运行才有值
        下面的逻辑只是在拼接网络结构
        """
        assert x.shape[1].value == 784
        x_images = tf.reshape(x, [-1, 28, 28, 1])

        # 拼接卷积层
        conv_in = x_images

        for i, conv_filter_shape in enumerate(self.conv_filter_shape_list):
            conv_op_name = 'conv%s' % (i + 1)
            conv_out = conv_op(conv_in, filters_shape=conv_filter_shape, variable_scope_name=conv_op_name)
            print('##', conv_op_name, conv_out)  # 打印卷积形状
            conv_in = conv_out  # 当前卷积的输出作为下层卷积的输入

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
                fc_out = tf.nn.dropout(fc_out, keep_prob)  # Dropout

                # 当前层的输出作为下一层的输入
                input_unit_num = fc_unit_num
                fc_in = fc_out
            print('##', fc_op_name, fc_out)

        # 模型前向计算的输出
        self.output = fc_out
        return fc_out
