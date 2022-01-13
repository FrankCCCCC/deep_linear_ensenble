import os
from typing import List, Tuple, Union
import pickle
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend
from tensorflow.keras.applications.vgg19 import VGG19

def finite_CNN(input_shape: Tuple[int, ...], classes: int, layer_num: int, kernel_size: Tuple[int, int], conv_block: int, channel: int, classifier_activation: str):
    strides = (2, 2)
    padding = 'same'
    pool_size = (2, 2)
    # kernel_size = (3, 3)

    # weight_init_std = np.sqrt(2)
    bias_init_std = np.sqrt(0.01)

    # weight_init_std = np.sqrt(2) / np.sqrt(channel)
    # dense_weight_init_std = np.sqrt(2) / np.sqrt(classes)

    weight_init = tf.keras.initializers.HeNormal()
    # weight_init = tf.keras.initializers.Orthogonal(gain=1.0)
    # weight_init = tf.random_normal_initializer(mean=0.0, stddev=weight_init_std)
    # dense_weight_init = tf.random_normal_initializer(mean=0.0, stddev=dense_weight_init_std)
    bias_init = tf.random_normal_initializer(mean=0.0, stddev=bias_init_std)

    inputs = layers.Input(shape=input_shape)
    for j in range(conv_block):
        x = layers.Conv2D(channel, kernel_size, strides=(1, 1), activation=None, padding=padding,
                        kernel_initializer=weight_init, bias_initializer=bias_init)(inputs)
        x = layers.LayerNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        # x = layers.Conv2D(channel, kernel_size, strides=strides, activation=None, padding=padding,
        #                   kernel_initializer=weight_init, bias_initializer=bias_init)(x)
        # x = layers.LayerNormalization()(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        # x = tf.keras.activations.sigmoid(x)

    x = layers.AveragePooling2D(pool_size=pool_size, strides=None, padding=padding)(x)

    # x = layers.Conv2D(channel, kernel_size, strides=strides, activation=None, padding=padding,
    #                     kernel_initializer=weight_init, bias_initializer=bias_init)(x)
    # x = layers.LayerNormalization()(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)

    # x = layers.MaxPool2D(pool_size=pool_size, strides=None, padding=padding)(x)
    # x = layers.LayerNormalization()(x)

    for i in range(layer_num - 1):
        for j in range(conv_block):
            # pass
            x = layers.Conv2D(channel, kernel_size, strides=(1, 1), activation=None, padding=padding,
                            kernel_initializer=weight_init, bias_initializer=bias_init)(x)
            # x = layers.Conv2D(channel, kernel_size, strides=strides, activation='relu', padding=padding,
            #                 kernel_initializer=weight_init, bias_initializer=bias_init)(x)
            x = layers.LayerNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
            # x = tf.keras.activations.sigmoid(x)

        x = layers.AveragePooling2D(pool_size=pool_size, strides=None, padding=padding)(x)
        # x = layers.LayerNormalization()(x)
        # x = layers.MaxPool2D(pool_size=pool_size, strides=None, padding=padding)(x)

    x = layers.Flatten()(x)
    # x = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)(x)
    # x = layers.Dense(channel,  activation=None, 
    #                  kernel_initializer=weight_init, bias_initializer=bias_init)(x)
    # x = layers.LayerNormalization()(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = tf.keras.activations.sigmoid(x)
    # x = layers.Dense(classes, activation=classifier_activation, 
    #                  kernel_initializer=weight_init, bias_initializer=bias_init)(x)
    # x = layers.Dense(classes, activation=None, 
    #                  kernel_initializer=weight_init, bias_initializer=bias_init)(x)

    return Model(inputs, x, name='Finite_CNN')

class DeepLinearRegr():
    def __init__():
        pass
    

class ModelMgr():
    VGG19_MODEL = 'VGG19'
    FINITE_CNN_MODEL = 'FINITE_CNN'

    METRIC_NAME = 'accuracy'
    LOSS_NAME = 'loss'

    METRIC_TYPE = 'Categorical Accuracy'
    LOSS_TYPE = 'MSE'

    CKPT_MONITOR = f'val_{METRIC_NAME}'

    def __init__(self, input_shape: Tuple[int, ...], classes: int, model_type: str, layer_num: int=8, width: int=512, kernel_size: Tuple[int, int]=(3, 3), conv_block: int=1, classifier_activation: str='softmax', is_freeze: bool=True):
        self.input_shape = input_shape
        self.classes = classes
        self.model_type = model_type
        self.layer_num = layer_num
        self.width = width
        self.kernel_size = kernel_size
        self.conv_block = conv_block
        self.classifier_activation = classifier_activation
        self.is_freeze = is_freeze

    @staticmethod
    def print_layers_freeze(model):
        layers_info = "Is trainable layers - "
        for i, layer in enumerate(model.layers):
            layers_info += f"[{i}]: {layer.trainable}"
        print(layers_info)

    @staticmethod
    def get_ckpt_callback(checkpoint_path: str):
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                            save_weights_only=False,
                                                            monitor=ModelMgr.CKPT_MONITOR,
                                                            save_best_only=True,
                                                            mode='max',
                                                            verbose=1)
        return ckpt_callback

    @staticmethod
    def compile(model, learning_rate: float=0.001):
        model.compile(
                    #   optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
                    optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                    loss=tf.keras.losses.MeanSquaredError(name=ModelMgr.LOSS_NAME),
                    #   loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=[tf.keras.metrics.CategoricalAccuracy(name=ModelMgr.METRIC_NAME)]
                    #   metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
                    )

        # model.compile(optimizer='adam',
        #               loss='sparse_categorical_crossentropy',
        #               metrics=['accuracy'])

        return model

    def get_model_arch(self):
        if self.model_type == ModelMgr.VGG19_MODEL:
            model = VGG19(weights=None, input_shape=self.input_shape, classes=self.classes, classifier_activation=self.classifier_activation)
        elif self.model_type == ModelMgr.FINITE_CNN_MODEL:
            model = finite_CNN(input_shape=self.input_shape, classes=self.classes, layer_num=self.layer_num, channel=self.width, kernel_size=self.kernel_size, conv_block=self.conv_block, classifier_activation=self.classifier_activation)
        else:
            raise ValueError(f'No such model called {self.model_type}')
        return model

    def get_model(self):
        model = self.get_model_arch()
        
        model.summary()
        print(f"Number of layers: {len(model.layers)}")
        if self.is_freeze:
            for layer in model.layers:
                layer.trainable = False
            model.layers[-1].trainable = True

        ModelMgr.print_layers_freeze(model=model)

        return model

    def get_ensemble(self, n_model: int):
        ensembles = []
        for i in range(n_model):
            ensembles.append(self.get_model())

        ensembles[0].summary()

        return ensembles