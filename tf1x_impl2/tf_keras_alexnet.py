import tensorflow as tf

from tensorflow.keras import layers

if tf.__version__.startswith('1.'):
    print('## TensorFlow Version 1x')
    # 在 tensorflow 1x examples 中，mnist 图片存储形状为 (None, 784)
    from tf1x_common_utils import mnist_load_data

    mnist = mnist_load_data()
    (X_train, y_train) = mnist.train.images, mnist.train.labels
    (X_test, y_test) = mnist.test.images, mnist.test.labels
elif tf.__version__.startswith('2.'):
    print('## TensorFlow Version 2x')
    # 在 keras 中， mnist 图片存储形状为 (None, 28, 28)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train / 255., X_test / 255.
    # (None, 28, 28) -> (None, 784)
    import numpy as np

    X_train = np.reshape(X_train, (-1, 784))
    X_test = np.reshape(X_test, (-1, 784))

    # 加载上来的标签是数字，不是 one_hot
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

else:
    raise Exception('ha ha !!')


def create_alexnet_model():
    return tf.keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(784,)),
        layers.Conv2D(64, 3, padding='same', data_format='channels_last', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_last'),
        layers.Conv2D(128, 3, padding='same', data_format='channels_last', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_last'),
        layers.Conv2D(256, 3, padding='same', data_format='channels_last', activation='relu'),
        layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_last'),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])


class AlexNetModel(tf.keras.Model):
    def __init__(self):
        """ 通过测试运行，可以发现该类在 TensorFlow 1x, TensorFlow 2x 中都可以运行
        """
        super(AlexNetModel, self).__init__(name='alexnet_model')

        print('#### 在 tensorflow {} 中使用 tf.keras.Model 对象的 call模式 训练模型'.format(tf.__version__))

        self.reshape = layers.Reshape((28, 28, 1), input_shape=(784,))
        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu', data_format='channels_last')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu', data_format='channels_last')
        self.conv3 = layers.Conv2D(64, 3, padding='same', activation='relu', data_format='channels_last')

        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', data_format='channels_last')

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)


def alexnet_train():
    # model = create_alexnet_model()
    model = AlexNetModel()

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-3),
                  metrics=['accuracy'])

    # 训练模型
    model.fit(X_train, y_train, batch_size=50, epochs=1)
    model.summary()
