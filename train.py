# %%
from typing import List, Tuple

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, backend
from tensorflow.keras.applications.vgg19 import VGG19
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# from dataset import Dataset
from scalablerunner.taskrunner import TaskRunner

def init_env(gpu_id: str):
    tf.keras.backend.set_floatx('float32')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    # os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'

VGG19_MODEL = 'VGG19'
FINITE_CNN_MODEL = 'FINITE_CNN'

def finite_CNN(input_shape: Tuple[int, ...], classes: int, classifier_activation: str='softmax'):
    layer_num = 8
    channel = 512
    # weight_init_std = np.sqrt(2)
    bias_init_std = np.sqrt(0.01)

    # weight_init_std = np.sqrt(2) / np.sqrt(channel)
    # dense_weight_init_std = np.sqrt(2) / np.sqrt(classes)

    weight_init = tf.keras.initializers.HeNormal()
    # weight_init = tf.random_normal_initializer(mean=0.0, stddev=weight_init_std)
    # dense_weight_init = tf.random_normal_initializer(mean=0.0, stddev=dense_weight_init_std)
    bias_init = tf.random_normal_initializer(mean=0.0, stddev=bias_init_std)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(channel, (3, 3), strides=(1, 1), activation='relu', padding='same',
                     kernel_initializer=weight_init, bias_initializer=bias_init)(inputs)

    for i in range(layer_num - 1):    
        x = layers.Conv2D(channel, (3, 3), strides=(1, 1), activation='relu', padding='same',
                          kernel_initializer=weight_init, bias_initializer=bias_init)(x)

    # x = layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
    #                   kernel_initializer=weight_init, bias_initializer=bias_init)(x)

    # x = layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)(x)
    x = layers.Dense(classes, activation=classifier_activation, 
                     kernel_initializer=weight_init, bias_initializer=bias_init)(x)

    return Model(inputs, x, name='Finite_CNN')

def ensemble_model(input_shape: Tuple[int, ...], classes: int, model_type: str, classifier_activation: str='softmax'):
    if model_type == VGG19_MODEL:
        model = VGG19(weights=None, input_shape=input_shape, classes=classes, classifier_activation=classifier_activation)
    elif model_type == FINITE_CNN_MODEL:
        model = finite_CNN(input_shape=input_shape, classes=classes, classifier_activation=classifier_activation)
    else:
        raise ValueError(f'No such model called {model_type}')
    
    model.summary()
    print(f"{len(model.layers)}")
    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True

    for i, layer in enumerate(model.layers):
        print(f"[{i}]: {layer.trainable}")

    return model

def ensemble(n_model: int, input_shape: Tuple[int, ...], classes: int):
    ensembles = []
    for i in range(n_model):
        ensembles.append(ensemble_model(input_shape=input_shape, classes=classes))

    ensemble[0].summary()

    return ensembles

def compile(model):
    model.compile(
                #   optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                  loss=tf.keras.losses.MeanSquaredError(name='MSE'),
                #   loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')]
                #   metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
                )

    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    return model

def _normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return (x / 255) * 2 - 1

def get_CIFAR10(sel_label: List[int]=None, is_onehot: bool=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if sel_label is not None:
        # Train
        conds = np.squeeze(np.isin(y_train, sel_label))
        # print(f"y_train.shape: {y_train.shape} | x_train.shape: {x_train.shape} | conds.shape: {conds.shape}")
        y_train = y_train[conds, ...]
        x_train = x_train[conds, ...]

        # Test
        conds = np.squeeze(np.isin(y_test, sel_label))
        # print(f"y_test.shape: {y_test.shape} | x_test.shape: {x_test.shape} | conds.shape: {conds.shape}")
        y_test = y_test[conds, ...]
        x_test = x_test[conds, ...]

    total_class_num = np.max(y_train) + 1

    print(f"x_train.shape: {x_train.shape} | {x_train[0, 0, 0]}")
    x_train = _normalize(x_train)
    x_test = _normalize(x_test)
    print(f"x_train.shape: {x_train.shape} | {x_train[0, 0, 0]}")

    print(f"y_train.shape: {y_train.shape} | {y_train[:5]}")
    if is_onehot:
        y_train = tf.one_hot(tf.squeeze(y_train), total_class_num)
        y_test = tf.one_hot(tf.squeeze(y_test), total_class_num)
    print(f"y_train.shape: {y_train.shape} | {y_train[:5]}")

    for img, label in zip(x_train[:10], y_train[:10]):
        plt.imshow(img)
        plt.show()
        print(f"Label: {label}")

    return (x_train, y_train), (x_test, y_test)

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()

    def get_gradient_func(model):
        grads = backend.gradients(model.total_loss, model.trainable_weights)
        inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
        func = backend.function(inputs, grads)
        return func

    def on_train_batch_end(self, batch, logs=None):
        # get_gradient = self.get_gradient_func(model)
        # grads = get_gradient([train_images, train_labels, np.ones(len(train_labels))])
        # epoch_gradient.append(grads)
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

def train(model, x_train, y_train, x_test, y_test, batch_size, epoch):
    model = compile(model)
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose="auto")
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    y_pred = model.predict(x_test)

    return model, y_pred, loss, acc

def train_loop(model, x_train, y_train, x_test, y_test, batch_size):
    TRAIN_BUF = 100000
    seed = 42
    x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(TRAIN_BUF, seed).batch(batch_size, drop_remainder=True)
    y_train = tf.data.Dataset.from_tensor_slices(y_train).shuffle(TRAIN_BUF, seed).batch(batch_size, drop_remainder=True)

    x_test = tf.data.Dataset.from_tensor_slices(x_test).shuffle(TRAIN_BUF, seed).batch(batch_size, drop_remainder=True)
    y_test = tf.data.Dataset.from_tensor_slices(y_test).shuffle(TRAIN_BUF, seed).batch(batch_size, drop_remainder=True)


    model = compile(model)
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=1, verbose="auto", callbacks=[CustomCallback])
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    y_pred = model.predict(x_test)

    return model, y_pred, loss, acc

def run(config: dict) -> None: 
    """
    A simple function for running specific config.
    """
    tr = TaskRunner(config=config)
    tr.output_log(file_name='logs/taskrunner.log')
    tr.run()

# %%
if __name__ == '__main__':
    sel_label = None
    classifier_activation = 'softmax'
    is_onehot = True
    batch_size = None
    epoch = 10

    if sel_label is not None:
        classes = len(sel_label)
        # if classes == 2:
        #     is_onehot = True
    else:
        classes = 10

    init_env('2')
    (x_train, y_train), (x_test, y_test) = get_CIFAR10(sel_label=sel_label, is_onehot=is_onehot)
    print(f"y_train.shape OneHot: {y_train.shape}")
    model = ensemble_model((32, 32, 3), classes, model_type=FINITE_CNN_MODEL, classifier_activation=classifier_activation)
    train(model, x_train, y_train, x_test, y_test, batch_size, epoch)
# %%
"""
Flatten:
    Channel: 64
    ALL: 22s 14ms/step - loss: 1.4347 - accuracy: 0.4875
    Freeze: 12s 7ms/step - loss: 3.0739 - accuracy: 0.3324

    Channel: 128
    30s 19ms/step - loss: 0.0776 - acc: 0.3575 | 3s - loss: 0.0725 - acc: 0.4209

    Channel: 256
    63s 40ms/step - loss: 0.0994 - acc: 0.3030 | 5s - loss: 0.0698 - acc: 0.4379

AvgPooling:

"""
# %%
