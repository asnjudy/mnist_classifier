import tensorflow as tf
from tensorflow.keras import layers
from tf1x_common_utils import mnist_load_data

mnist = mnist_load_data()

X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels


def tf_keras_model_fn(features, labels, mode, params):
    model = tf.keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(784,)),
        layers.Conv2D(64, 3, padding='same', data_format='channels_last', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_last'),
        layers.Conv2D(128, 3, padding='same', data_format='channels_last', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_last'),
        layers.Conv2D(256, 3, padding='same', data_format='channels_last', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_last'),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(10)  # 在交叉熵损失中做了 softmax 概率归一化，所以此次就不用 softmax 激活了
    ])

    if mode == tf.estimator.ModeKeys.PREDICT:
        output = model(features, training=False)
        predicted_classes = tf.argmax(output, axis=1)
        return tf.estimator.EstimatorSpec(mode, predictions=predicted_classes)

    output = model(features, training=True)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output)
    loss = tf.reduce_mean(cross_entropy)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # 选择优化算法很重要
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # 定义准确率
        predicted_classes = tf.argmax(output, axis=1)
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predicted_classes)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'accuracy': accuracy})


def tf_nn_model_fn(features, labels, mode, params):
    from tf1x_impl1.alexnet_model import AlexNetModel2

    x = features

    if mode == tf.estimator.ModeKeys.TRAIN:
        keep_prob = 0.5
    else:
        keep_prob = 1.0

    model = AlexNetModel2()
    output = model(x, keep_prob=keep_prob)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_classes = tf.argmax(output, axis=1)
        return tf.estimator.EstimatorSpec(mode, predictions=predicted_classes)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output)
    loss = tf.reduce_mean(cross_entropy)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # 选择优化算法很重要
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # 定义准确率
        predicted_classes = tf.argmax(output, axis=1)
        accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predicted_classes)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'accuracy': accuracy})


def alexnet_train(model_fn):
    # 创建 Estimator 对象
    config = tf.estimator.RunConfig(model_dir='./logs', log_step_count_steps=30)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=X_train,
                                                        y=y_train,
                                                        batch_size=128,
                                                        shuffle=True,
                                                        num_epochs=10)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=X_test, y=y_test, batch_size=128, shuffle=False)

    estimator.train(input_fn=train_input_fn, steps=1000)

    estimator.evaluate(input_fn=eval_input_fn, steps=1200)
