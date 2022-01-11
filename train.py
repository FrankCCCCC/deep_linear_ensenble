# %%
import os
from typing import List, Tuple
import pickle
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import update
from tensorflow.python.types.core import Value
from tqdm import tqdm

# from dataset import Dataset
from scalablerunner.taskrunner import TaskRunner

def init_env(gpu_id: str):
    tf.keras.backend.set_floatx('float32')
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    # os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    
def check_make_dir(path: str, is_delete_if_exist: bool=False):
    """
    Check whether the directory exist or not and make the directory if it doesn't exist
    """
    if os.path.exists(path):
        if is_delete_if_exist:
            os.rmdir(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return path

def save_pkl(obj: object, file: str):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def load_pkl(file: str):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
        return obj

SHARE_DISK = '/opt/shared-disk2/sychou/ensemble'

def finite_CNN(input_shape: Tuple[int, ...], classes: int, layer_num: int, conv_block: int, channel: int, classifier_activation: str):
    # strides = (1, 1)
    strides = (2, 2)
    padding = 'same'
    pool_size = (2, 2)
    kernel_size = (3, 3)

    # weight_init_std = np.sqrt(2)
    bias_init_std = np.sqrt(0.01)

    # weight_init_std = np.sqrt(2) / np.sqrt(channel)
    # dense_weight_init_std = np.sqrt(2) / np.sqrt(classes)

    # weight_init = tf.keras.initializers.HeNormal()
    weight_init = tf.keras.initializers.Orthogonal(gain=1.0)
    # weight_init = tf.random_normal_initializer(mean=0.0, stddev=weight_init_std)
    # dense_weight_init = tf.random_normal_initializer(mean=0.0, stddev=dense_weight_init_std)
    bias_init = tf.random_normal_initializer(mean=0.0, stddev=bias_init_std)

    inputs = layers.Input(shape=input_shape)
    for j in range(conv_block):
        x = layers.Conv2D(channel, (3, 3), strides=(1, 1), activation='relu', padding=padding,
                          kernel_initializer=weight_init, bias_initializer=bias_init)(inputs)
        # x = layers.Conv2D(channel, kernel_size, strides=strides, activation='relu', padding=padding,
        #                   kernel_initializer=weight_init, bias_initializer=bias_init)(x)
    # x = layers.LayerNormalization()(x)
    # x = tf.keras.activations.sigmoid(x)
    x = layers.AveragePooling2D(pool_size=pool_size, strides=None, padding=padding)(x)
    # x = layers.MaxPool2D(pool_size=pool_size, strides=None, padding=padding)(x)
    # x = layers.LayerNormalization()(x)

    for i in range(layer_num - 1):
        for j in range(conv_block):
            x = layers.Conv2D(channel, (3, 3), strides=(1, 1), activation='relu', padding=padding,
                              kernel_initializer=weight_init, bias_initializer=bias_init)(x)
            # x = layers.Conv2D(channel, kernel_size, strides=strides, activation='relu', padding=padding,
            #                 kernel_initializer=weight_init, bias_initializer=bias_init)(x)
        # x = layers.LayerNormalization()(x)
        # x = tf.keras.activations.sigmoid(x)
        x = layers.AveragePooling2D(pool_size=pool_size, strides=None, padding=padding)(x)
        # x = layers.LayerNormalization()(x)
        # x = layers.MaxPool2D(pool_size=pool_size, strides=None, padding=padding)(x)

    # x = layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
    #                   kernel_initializer=weight_init, bias_initializer=bias_init)(x)

    x = layers.Flatten()(x)
    # x = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)(x)
    x = layers.Dense(channel,  activation='relu', 
                     kernel_initializer=weight_init, bias_initializer=bias_init)(x)
    # x = layers.LayerNormalization()(x)
    # x = tf.keras.activations.sigmoid(x)
    x = layers.Dense(classes, activation=classifier_activation, 
                     kernel_initializer=weight_init, bias_initializer=bias_init)(x)
    # x = layers.Dense(classes, activation=None, 
    #                  kernel_initializer=weight_init, bias_initializer=bias_init)(x)

    return Model(inputs, x, name='Finite_CNN')

class ModelMgr():
    VGG19_MODEL = 'VGG19'
    FINITE_CNN_MODEL = 'FINITE_CNN'

    METRIC_NAME = 'accuracy'
    LOSS_NAME = 'loss'

    METRIC_TYPE = 'Categorical Accuracy'
    LOSS_TYPE = 'MSE'

    CKPT_MONITOR = f'val_{METRIC_NAME}'

    def __init__(self, input_shape: Tuple[int, ...], classes: int, model_type: str, layer_num: int=8, width: int=512, conv_block: int=1, classifier_activation: str='softmax', is_freeze: bool=True):
        self.input_shape = input_shape
        self.classes = classes
        self.model_type = model_type
        self.layer_num = layer_num
        self.width = width
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
            model = finite_CNN(input_shape=self.input_shape, classes=self.classes, layer_num=self.layer_num, channel=self.width, conv_block=self.conv_block, classifier_activation=self.classifier_activation)
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

class RecordMgr():
    CKPT_PATH = 'checkpoints'
    RESULT_PATH = 'results'
    CKPT_ENSEMBLE_PREFIX = 'ensemble_'
    CKPT_NAME = 'ckpt'
    SAVED_MOEDL_NAME = 'saved_model.h5'
    RECORD_DIR_NAME = 'record'
    RECORD_NAME = 'record.pkl'
    CONFIG_NAME = 'config.json'
    FIGURE_DIR_NAME = 'figure'

    def __init__(self, base_path: str):
        self.base_path = base_path
        check_make_dir(path=base_path)
    
    @staticmethod
    def get_ensemble_model_path(base_path: str, id: int):
        dir_name = f'{RecordMgr.CKPT_ENSEMBLE_PREFIX}{id}'
        model_path = os.path.join(base_path, dir_name)
        check_make_dir(path=model_path)
        return model_path

    @staticmethod
    def get_ensemble_model_RecordMgr(base_path: str, id: int):
        record_mgr = RecordMgr(base_path=RecordMgr.get_ensemble_model_path(base_path=base_path, id=id))
        return record_mgr

    def get_model_ckpt_path(self):
        ckpt_path = os.path.join(self.base_path, RecordMgr.CKPT_NAME)
        return ckpt_path

    def get_saved_model_path(self):
        saved_model_path = os.path.join(self.get_model_ckpt_path(), RecordMgr.SAVED_MOEDL_NAME)
        return saved_model_path

    def get_model_records_path(self):
        record_path = os.path.join(self.base_path, RecordMgr.RECORD_DIR_NAME)
        check_make_dir(path=record_path)
        print(f"record_path: {record_path}")
        return record_path

    def get_model_figures_path(self):
        figure_path = os.path.join(self.base_path, RecordMgr.FIGURE_DIR_NAME)
        check_make_dir(path=figure_path)
        print(f"figure_path: {figure_path}")
        return figure_path

    def get_model_history_file(self):
        return os.path.join(self.get_model_records_path(), RecordMgr.RECORD_NAME)

    def get_model_figure_file(self, fig_name: str):
        if fig_name is None:
            raise ValueError(f"The argument 'fig_name' shouldn't be {type(fig_name)}, it should be a string")
        return os.path.join(self.get_model_figures_path(), fig_name)

    def get_model_config_file(self):
        return os.path.join(self.get_model_records_path(), RecordMgr.CONFIG_NAME)

    def save_config(self, config: dict):
        config_file = self.get_model_config_file()
        with open(config_file, 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)

    def save_history(self, history: object):
        history_path = self.get_model_history_file()
        save_pkl(obj=history, file=history_path)

    def load_history(self):
        history_path = self.get_model_history_file()
        return load_pkl(file=history_path)

    def save_model(self, model):
        tf.keras.models.save_model(model, self.get_saved_model_path())

    def load_model(self):
        model = tf.keras.models.load_model(self.get_saved_model_path())
        return model

    def load_ensemble(self, num_model: int, *args, **kwargs):
        model_list = []
        print(f"Loading Ensemble...")
        for i in range(num_model):
            model = self.load_model(*args, **kwargs)
            model_list.append(model)
        
        return model_list

def _normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return ((x / 255) * 2 - 1) * 2

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

    print(f"y_train.shape: {y_train.shape} | {y_train[0]}")
    print(f"y_test.shape: {y_test.shape} | {y_test[0]}")
    if is_onehot:
        y_train = tf.one_hot(tf.squeeze(y_train), total_class_num)
        y_test = tf.one_hot(tf.squeeze(y_test), total_class_num)
    print(f"y_train.shape: {y_train.shape} | {y_train[0]}")
    print(f"y_test.shape: {y_test.shape} | {y_test[0]}")

    # for img, label in zip(x_train[:10], y_train[:10]):
    #     plt.imshow(img)
    #     plt.show()
    #     print(f"Label: {label}")

    return (x_train, y_train), (x_test, y_test)

# class CustomCallback(tf.keras.callbacks.Callback):
#     def __init__(self):
#         super(CustomCallback, self).__init__()

#     def get_gradient_func(model):
#         grads = backend.gradients(model.total_loss, model.trainable_weights)
#         inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
#         func = backend.function(inputs, grads)
#         return func

#     def on_train_batch_end(self, batch, logs=None):
#         # get_gradient = self.get_gradient_func(model)
#         # grads = get_gradient([train_images, train_labels, np.ones(len(train_labels))])
#         # epoch_gradient.append(grads)
#         keys = list(logs.keys())
#         print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

def train(model: tf.keras.Model, x_train, y_train, x_test, y_test, batch_size, epoch, lr: float=0.001, base_path: str=None, config: dict=None):
    model = ModelMgr.compile(model, learning_rate=lr)
    record_mgr = RecordMgr(base_path=base_path)
    visulize_mgr = VisulaizeMgr(base_path=base_path)

    if config is not None:
        record_mgr.save_config(config=config)

    if base_path is not None:
        ckpt_callback = ModelMgr.get_ckpt_callback(checkpoint_path=record_mgr.get_saved_model_path())        
        history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose="auto",
                  validation_data=(x_test, y_test), validation_freq=1, callbacks=[ckpt_callback])
        record_mgr.save_history(history=history.history)
    else:
        # No checkpoint
        history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose="auto",
                  validation_data=(x_test, y_test), validation_freq=1)

    visulize_mgr.plot_history(history=record_mgr.load_history())

    model, loss, acc = evaluate_model(model=model, x_test=x_test, y_test=y_test)
    y_pred = model.predict(x_test)

    return model, y_pred, loss, acc

def evaluate_model(model: tf.keras.Model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    return model, loss, acc

def evaluate_ensemble(model_list: List[tf.keras.Model], x_test, y_test, eval_non_act: bool=False):
    loss_metric = tf.keras.losses.MeanSquaredError()
    eval_metric = tf.keras.metrics.CategoricalAccuracy()
    
    # Ensemble
    y_pred_ensemble = []
    eval_ensemble = []
    loss_ensemble = []
    print(f"Evaluating Ensemble...")
    for i, model in enumerate(tqdm(model_list)):
        loss_metric_ensemble = tf.keras.losses.MeanSquaredError()
        eval_metric_ensemble = tf.keras.metrics.CategoricalAccuracy()
        y_pred, layer_inputs, layer_outputs, layer_weights, layer_outputs_non_act = model_predict(model=model, x_test=x_test)
        # print(f"y_pred_ensemble.shape: {y_pred_ensemble.shape} | y_pred.shape: {y_pred.shape}")
        # if eval_non_act:
        #     y_pred_ensemble.append(layer_outputs_non_act[-1])
        # else:
        y_pred_ensemble.append(y_pred)
        
        eval_metric_ensemble.update_state(y_test, y_pred)
        pred_acc = eval_metric_ensemble.result().numpy()
        pred_loss = loss_metric(y_test, y_pred)
        eval_ensemble.append(pred_acc)
        loss_ensemble.append(pred_loss)
        print(f"{i} - Acc: {pred_acc} | Loss: {pred_loss}")

        # print(f"y_pred.shape: {y_pred.shape}")

    print(f"Max Acc: {max(eval_ensemble)} | Min Loss: {min(loss_ensemble)}")

    # Mean prediction
    y_pred_ensemble = np.stack(y_pred_ensemble, axis=0)
    print(f"y_pred_ensemble.shape: {y_pred_ensemble.shape}")
    y_pred_mean = np.mean(y_pred_ensemble, axis=0)
    # if eval_non_act:
    #     y_pred_mean = tf.nn.softmax(y_pred_mean)
    print(f"y_pred_mean.shape: {y_pred_mean.shape}")
    eval_metric.update_state(y_test, y_pred_mean)
    acc = eval_metric.result().numpy()
    loss = loss_metric(y_test, y_pred_mean)

    print(f"{len(model_list)} Ensemble Eval - Acc: {acc} | Loss: {loss}")
    
    return model_list, loss, acc

def model_predict(model: tf.keras.Model, x_test, is_get_layer_input: bool=False, is_get_layer_output: bool=False, is_get_layer_output_non_act: bool=False, is_get_layer_weight: bool=False):
    y_pred = model.predict(x_test)

    layer_inputs = []
    if is_get_layer_input:
        for layer in model.layers:
            layer_inputs.append(layer.input)

    layer_outputs = []
    if is_get_layer_output:
        for layer in model.layers:
            layer_outputs.append(layer.output)

    layer_weights = []
    if is_get_layer_weight:
        for layer in model.layers:
            layer_weights.append(layer.get_weights())

    layer_outputs_non_act = []
    if is_get_layer_output_non_act:
        for layer in model.layers:
            # print(f"layer.weights type: {layer.weights} | layer.input.shape: {layer.input.shape}")
            print(f"layer.weights.shape: {layer.weights[0].shape} | layer.input.shape: {layer.input.shape}")
            output_non_act = tf.matmul(layer.weights[0], layer.input)
            layer_outputs_non_act.append(output_non_act)

    return y_pred, layer_inputs, layer_outputs, layer_weights, layer_outputs_non_act

def train_a_model(gpu_id: str, epoch: int, id: int=0, sel_label: List[int]=None, batch_size: int=32, layer_num: int=8, width: int=512, conv_block: int=1,
                  model_type: str=ModelMgr.FINITE_CNN_MODEL, classifier_activation: str='softmax', is_freeze: bool=True, lr: float=0.001, base_path: str=None):
    is_onehot = True
    config = locals()

    if sel_label is not None:
        classes = len(sel_label)
        if classes == 2:
            is_onehot = False
    else:
        classes = 10

    if id is not None:
        # Ensemble directory
        exp_path = RecordMgr.get_ensemble_model_path(base_path=base_path, id=id)
    else:
        # Hyperparameter naming
        exp_path = os.path.join(base_path, f'{model_type}_ln-{layer_num}_w-{width}_cb-{conv_block}_act-{classifier_activation}_ep-{epoch}_bs-{batch_size}_lr-{lr}')
    model_mgr = ModelMgr(input_shape=(32, 32, 3), classes=classes, model_type=model_type, layer_num=layer_num, width=width, conv_block=conv_block, classifier_activation=classifier_activation, is_freeze=is_freeze)

    init_env(gpu_id=gpu_id)
    (x_train, y_train), (x_test, y_test) = get_CIFAR10(sel_label=sel_label, is_onehot=is_onehot)
    # print(f"y_train.shape OneHot: {y_train.shape}")
    model = model_mgr.get_model()
    model, y_pred, loss, acc = train(model, x_train, y_train, x_test, y_test, batch_size, epoch, lr, base_path=exp_path, config=config)

class VisulaizeMgr():
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.recod_mgr = RecordMgr(base_path=base_path)
        # self.visualize_mgr = VisulaizeMgr(base_path=base_path)

    def plot_history(self, history, is_show: bool=True, is_save: bool=True):
        metric_name = ModelMgr.METRIC_NAME
        loss_name = ModelMgr.LOSS_NAME
        train_label_name = 'Train'
        test_label_name = 'Test'

        # Acc
        train_acc = history[metric_name]
        val_acc = history[f'val_{metric_name}']
        plt.figure(figsize=(8, 6), dpi=130)
        plt.plot(train_acc, label=train_label_name)
        plt.plot(val_acc, label=test_label_name)
        plt.title(f'Model accuracy (Val Best: {round(np.max(val_acc), 4)})')
        plt.ylabel(f'Accuracy ({ModelMgr.METRIC_TYPE})')
        plt.xlabel('Epoch')
        plt.legend()

        if is_save:
            plt.savefig(self.recod_mgr.get_model_figure_file(fig_name="acc.png"))
        if is_show:
            plt.show()

        # Loss
        train_loss = history[loss_name]
        val_loss = history[f'val_{loss_name}']
        plt.figure(figsize=(8, 6), dpi=130)
        plt.plot(train_loss, label=train_label_name)
        plt.plot(val_loss, label=test_label_name)
        plt.title(f'Model loss (Val Best: {round(np.max(val_loss), 4)})')
        plt.ylabel(f'Loss ({ModelMgr.LOSS_TYPE})')
        plt.xlabel('Epoch')
        plt.legend()

        if is_save:
            plt.savefig(self.recod_mgr.get_model_figure_file(fig_name="loss.png"))
        if is_show:
            plt.show()

def get_model_infos(model):
    layer_weights = []
    for layer in model.layers:
        w = layer.get_weights()
        print(f"size: {len(w)}| w: {w}")
        layer_weights.append(w)
# %%
if __name__ == '__main__':
    # init_env('1')
    train_a_model(gpu_id='0', layer_num=5, width=1024, conv_block=1, epoch=30, id=0, model_type=ModelMgr.FINITE_CNN_MODEL, classifier_activation=None, is_freeze=True, base_path=RecordMgr.RESULT_PATH)

    record_mgr = RecordMgr.get_ensemble_model_RecordMgr(base_path=RecordMgr.RESULT_PATH, id=0)
    model = record_mgr.load_model()
    (x_train, y_train), (x_test, y_test) = get_CIFAR10(sel_label=None, is_onehot=True)
    evaluate_model(model=model, x_test=x_test, y_test=y_test)

    # model_list = record_mgr.load_ensemble(num_model=50, input_shape=(32, 32, 3), classes=10, model_type=ModelMgr.FINITE_CNN_MODEL, base_path=SHARE_DISK, classifier_activation=None)
    # model_list, loss, acc = evaluate_ensemble(model_list=model_list, x_test=x_test, y_test=y_test, eval_non_act=False)

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
Channel: 512
Freeze: 76s 49ms/step |  12s - loss: 0.0741 - accuracy: 0.4080

Channel: 1024
Freeze: 246s 157ms/step | loss: 0.0700 - accuracy: 0.4432
"""
# %%
