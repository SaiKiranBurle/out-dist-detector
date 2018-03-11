import os

import keras.backend as K
import numpy as np
from keras.datasets.cifar import load_batch
from keras.engine import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import get_file
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import densenet

pretrained_model_weights_path = 'weights/DenseNet-100-12-CIFAR10.h5'
out_dist_path = '/Users/sai/dev/datasets/tiny-imagenet/test'


def get_in_dist_train_data():
    """
    Loads a small batch of CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_batches = 1     # 10K, 10K
    num_train_samples = 10000 * num_batches

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    # Load only one of the 5 train batches
    for i in range(1, 1 + num_batches):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        # Since in-dist images should have 0 as label
        # y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)
    # Since in-dist images should have 0 as label
    # Number of test images is 10000
    y_test = np.zeros((10000,), dtype='uint8')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    return (x_train, y_train), (x_test, y_test)


def get_out_dist_train_data():
    img_gen = ImageDataGenerator()
    num_train = 1000    # 1K, 9K
    data_gen = img_gen.flow_from_directory(out_dist_path, target_size=(32, 32),
                                           batch_size=10000, seed=0)
    data = data_gen.next()
    return (data[0][:num_train], data[1][:num_train]), (data[0][num_train:], data[1][num_train:])


def get_model():
    base_model = densenet.DenseNet((32, 32, 3), depth=100, growth_rate=12, bottleneck=True, weights=None)
    base_model.load_weights(pretrained_model_weights_path, by_name=True)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = tf.gather(x, tf.nn.top_k(x, k=10).indices)

    # Use output to determine in/out dist
    x = Dense(units=30, activation='relu')(x)
    x = Dense(units=5, activation='relu')(x)
    x = Dense(units=10, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    # Use gradients to determine in/out
    # gradients = K.gradients(x, base_model.input)
    # gradients = Flatten()(gradients)
    # predictions = Dense(2, activation='softmax')(gradients)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == "__main__":
    (x_in_train, y_in_train), (x_in_test, y_in_test) = get_in_dist_train_data()
    (x_out_train, y_out_train), (x_out_test, y_out_test) = get_out_dist_train_data()

    # Concatenate in and out dist images
    x_train_raw = np.concatenate((x_in_train, x_out_train))
    y_train_raw = np.concatenate((y_in_train, y_out_train))

    x_test_raw = np.concatenate((x_in_test, x_out_test))
    y_test_raw = np.concatenate((y_in_test, y_out_test))

    # Pre-process inputs
    x_train = densenet.preprocess_input(x_train_raw)
    x_test = densenet.preprocess_input(x_test_raw)

    y_train = np_utils.to_categorical(y_train_raw, 2)
    y_test = np_utils.to_categorical(y_test_raw, 2)

    # Shuffle training data
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_test, y_test = unison_shuffled_copies(x_test, y_test)

    model = get_model()
    model.fit(x=x_train,
              y=y_train,
              batch_size=32,
              epochs=100,
              validation_data=(x_test, y_test))

    from IPython import embed
    embed()
