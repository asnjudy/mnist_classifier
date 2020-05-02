from enum import Enum, unique
import tensorflow as tf


def create_weight_variable(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)


def create_bias_variable(shape, name=None):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def conv2d__max_pool_3x3__norm(x, W, bias, name=None):
    # 卷积
    h_conv = tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME', name=name)
    h_conv = tf.nn.relu(h_conv + bias)
    # 池化
    h_pool = tf.nn.max_pool(h_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 局部响应归一化
    h_norm = tf.nn.lrn(h_pool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    return h_norm


def conv_op(x, filters_shape, variable_scope_name):
    filters_num = filters_shape[-1]
    with tf.variable_scope(variable_scope_name) as conv:
        conv_filters = create_weight_variable(shape=filters_shape, name="filters")
        conv_biases = create_bias_variable(shape=(filters_num, ), name='bias')
        conv_out = conv2d__max_pool_3x3__norm(x, W=conv_filters, bias=conv_biases, name='conv')
        return conv_out


def fc_op(x, input_unit_num, unit_num, variable_scope_name):
    """ 全连接层，与前一层进行全连接
    input_unit_num, 表示前面那层包含的神经元个数
    unit_num, 代表本层中包含的神经元个数

    W 形状 (input_unit_num, unit_num)
    b 形状 (unit_num,)
    """
    with tf.variable_scope(variable_scope_name) as fc:
        W = create_weight_variable(shape=(input_unit_num, unit_num), name='weights')
        b = create_bias_variable(shape=(unit_num, ), name='bias')

        fc_out = tf.nn.bias_add(tf.matmul(x, W), b)  # 全连接
        return fc_out


@unique
class ModelMode(Enum):
    TRIAN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'
