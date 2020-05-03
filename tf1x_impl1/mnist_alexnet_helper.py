import tensorflow as tf
from tf1x_impl1.alexnet_model import AlexNetModel
from tf1x_common_utils import mnist_load_data


class MNISTAlexNetModelHelper(object):
    def __init__(self):
        self.mnist = mnist_load_data()
        self.model_dir = './logs'
        self.model = AlexNetModel()

    def save_alex_net_model(self):
        pass

    def load_alex_net_model(self):
        pass

    def train(self, batch_size=128, num_steps=1000, log_per_steps=100):
        # 使用 tf.Session API进行训练
        # 训练完成后，要保持模型各层权重（**重要**）。predict时加载模型权重前向计算给出结果

        if batch_size is not None:
            self.batch_size = batch_size
        if num_steps is not None:
            self.num_steps = num_steps

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # sess.run([train_op], feed_dict={....})

    def train_and_evaluate(self, num_steps=1000, log_per_steps=100, eval_per_steps=100):
        # 训练时每隔多少步打印一下损失和准确度
        # 训练时每个多少步在测试集上跑一下准确度
        pass

    def predict(self, images):
        # 加载模型权重
        # 前向计算给出结果，用 tf.Session 跑算一下 output
        return _out
