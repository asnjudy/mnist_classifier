def tf1x_impl1_1():
    """ 实现方案1-1
        用 tf.nn（tensorflow 最底层 API）搭建模型，使用 tf.Session 训练模型
        搭建模型过程抽取为函数alexnet，训练过程抽取为函数alexnet_train 
    """
    from tf1x_impl1.func_tf_Session_run import alexnet_train

    alexnet_train()


def tf1x_impl1_2():
    """ 实现方案1-2
        用 tf.nn（tensorflow 最底层 API）搭建模型，使用 tf.Session 训练模型
        模型抽象为类 AlexNetModel，训练过程抽取为函数alexnet_train2
    """
    from tf1x_impl1.func_tf_Session_run import alexnet_train2, alexnet_train3

    alexnet_train3()


def tf1x_impl1_3():
    """ 实现方案1-3
        用 tf.nn（tensorflow 最底层 API）搭建模型，使用 tf.Session 训练模型
        模型抽象为类 AlexNetModel，定义模型辅助训练类 MNISTAlexNetModelHelper
    """
    # 模型辅助训练类 MNISTAlexNetModelHelper 涉及到模型保存和加载，待实现
    pass


def tf1x_impl2_1():
    """ 实现方案2
        纯 keras 搭建模型并训练
    """
    from tf1x_impl2.tf_keras_alexnet import alexnet_train

    alexnet_train()


def tf1x_impl3_1():
    """ 实现方案3-1
        用 keras 搭建模型，使用 tf.estimator.Estimator 训练
    """
    from tf1x_impl3.alexnet_tf_Estimator import tf_keras_model_fn, alexnet_train

    alexnet_train(tf_keras_model_fn)


def tf1x_impl3_2():
    """ 实现方案3-2
        用 tf.nn 搭建模型，使用 tf.estimator.Estimator 训练
    """
    from tf1x_impl3.alexnet_tf_Estimator import tf_nn_model_fn, alexnet_train

    alexnet_train(tf_nn_model_fn)


if __name__ == "__main__":
    # tf1x_impl1_2()
    tf1x_impl3_2()
