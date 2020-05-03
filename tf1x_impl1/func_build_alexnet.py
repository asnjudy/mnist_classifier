import tensorflow as tf
from tf1x_common_utils import *


def alexnet(x, keep_prob):
    def weight_variable(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def bias_variable(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    conv2d_strides = [1, 1, 1, 1]
    max_pool_ksize = [1, 3, 3, 1]
    max_pool_strides = [1, 2, 2, 1]

    # 转换为卷积的输入
    x_images = tf.reshape(x, shape=(-1, 28, 28, 1))

    # 第一个卷积层
    with tf.variable_scope('conv1') as conv1:
        filters = create_weight_variable(shape=(3, 3, 1, 64), name="filters")
        biases = create_bias_variable(shape=(64, ), name='biases')
        conv1_out = conv2d__max_pool_3x3__norm(x_images, W=filters, bias=biases, name='conv')

    # 第二个卷积层
    with tf.variable_scope('conv2') as conv2:
        filters = create_weight_variable(shape=(3, 3, 64, 128), name="filters")
        biases = create_bias_variable(shape=(128, ), name='biases')
        conv2_out = conv2d__max_pool_3x3__norm(conv1_out, W=filters, bias=biases, name='conv')

    # 第三个卷积层
    with tf.variable_scope('conv2') as conv2:
        filters = create_weight_variable(shape=(3, 3, 128, 256), name="filters")
        biases = create_bias_variable(shape=(256, ), name='biases')
        conv3_out = conv2d__max_pool_3x3__norm(conv2_out, W=filters, bias=biases, name='conv')

    # 全连接层
    with tf.variable_scope('fc1') as fc1:
        W = create_weight_variable(shape=(4 * 4 * 256, 1024), name="weight")
        b = create_bias_variable(shape=[1024], name="bias")
        # 第二卷积层池化后的结果拉平，作为全连接层的输入
        conv3_out_flat = tf.reshape(conv3_out, [-1, 4 * 4 * 256])
        fc1_out = tf.nn.relu(tf.matmul(conv3_out_flat, W) + b)
        fc1_out_drop = tf.nn.dropout(fc1_out, keep_prob)

    # 全连接层，最后一个分类层直接使用 softmax 交叉熵损失，不需要Relu激活（激活就错了，会导致梯度更新时损失一直降不下来）
    with tf.variable_scope('fc2') as fc2:
        W = create_weight_variable(shape=(1024, 10), name="weight")
        b = create_bias_variable(shape=[10], name="bias")
        fc2_out = tf.matmul(fc1_out_drop, W) + b

    output = fc2_out
    print('the alexnet model definition finished!')
    print('### output shape:', output.shape)
    return output
