# %%
import os
from typing import List, Tuple
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import update
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

VGG19_MODEL = 'VGG19'
FINITE_CNN_MODEL = 'FINITE_CNN'
CKPT_PATH = 'checkpoints'
RESULT_PATH = 'results'
CKPT_ENSEMBLE_PREFIX = 'ensemble_'
CKPT_NAME = 'ckpt'
RECORD_DIR_NAME = 'record'
RECORD_NAME = 'record.pkl'

SHARE_DISK = '/opt/shared-disk2/sychou/ensemble'

def finite_CNN(input_shape: Tuple[int, ...], classes: int, channel: int=512, classifier_activation: str='softmax'):
    layer_num = 8
    # channel = 512
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

def print_layers_freeze(model):
    layers_info = ""
    for i, layer in enumerate(model.layers):
        layers_info += f"[{i}]: {layer.trainable}"
    print(layers_info)

def get_model_arch(input_shape: Tuple[int, ...], classes: int, model_type: str, width: int=512, classifier_activation: str='softmax'):
    if model_type == VGG19_MODEL:
        model = VGG19(weights=None, input_shape=input_shape, classes=classes, classifier_activation=classifier_activation)
    elif model_type == FINITE_CNN_MODEL:
        model = finite_CNN(input_shape=input_shape, classes=classes, channel=width, classifier_activation=classifier_activation)
    else:
        raise ValueError(f'No such model called {model_type}')
    return model

def get_model(input_shape: Tuple[int, ...], classes: int, model_type: str, width: int=512, classifier_activation: str='softmax'):
    model = get_model_arch(input_shape=input_shape, classes=classes, model_type=model_type, width=width, classifier_activation=classifier_activation)
    
    model.summary()
    print(f"{len(model.layers)}")
    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True

    print_layers_freeze(model=model)

    return model

def get_ensemble(n_model: int, input_shape: Tuple[int, ...], classes: int, width: int=512):
    ensembles = []
    for i in range(n_model):
        ensembles.append(get_model(input_shape=input_shape, classes=classes, width=width))

    ensembles[0].summary()

    return ensembles

def get_model_path(base_path: str, id: int):
    dir_name = f'{CKPT_ENSEMBLE_PREFIX}{id}'
    exp_path = os.path.join(base_path, dir_name)
    check_make_dir(path=exp_path)
    return exp_path

def get_model_ckpt_path(base_path: str):
    exp_path = os.path.join(base_path, CKPT_NAME)
    return exp_path

def get_model_records_path(base_path: str):
    exp_path = os.path.join(base_path, RECORD_DIR_NAME)
    check_make_dir(path=exp_path)
    print(f"exp_path: {exp_path}")
    return exp_path

def get_model_history_file(base_path: str):
    return os.path.join(base_path, RECORD_NAME)

def save_history(history: object, base_path: str):
    epx_path = get_model_records_path(base_path=base_path)
    epx_path = get_model_history_file(base_path=epx_path)
    save_pkl(obj=history, file=epx_path)

def load_history(base_path: str):
    epx_path = get_model_records_path(base_path=base_path)
    epx_path = get_model_history_file(base_path=epx_path)
    return load_pkl(file=epx_path)

def get_ckpt_callback(checkpoint_path: str):
    check_make_dir(path=checkpoint_path)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=get_model_ckpt_path(checkpoint_path),
                                                        save_weights_only=True,
                                                        monitor='val_accuracy',
                                                        save_best_only=True,
                                                        mode='max',
                                                        verbose=1)
    return ckpt_callback

def compile(model):
    model.compile(
                #   optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
                  optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                  loss=tf.keras.losses.MeanSquaredError(name='MSE'),
                #   loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
                #   metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
                )

    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    return model

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

def train(model: tf.keras.Model, x_train, y_train, x_test, y_test, batch_size, epoch, base_path: str=None):
    model = compile(model)
    if base_path is not None:
        ckpt_callback = get_ckpt_callback(checkpoint_path=base_path)        
        history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose="auto",
                  validation_data=(x_test, y_test), validation_freq=1, callbacks=[ckpt_callback])
        save_history(history=history.history, base_path=base_path)
    else:
        # No checkpoint
        history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epoch, verbose="auto",
                  validation_data=(x_test, y_test), validation_freq=1)

    model, loss, acc = evaluate_model(model=model, x_test=x_test, y_test=y_test)
    y_pred = model.predict(x_test)

    return model, y_pred, loss, acc

def load_model(id: int, input_shape: Tuple[int, ...], classes: int, model_type: str, base_path: str=RESULT_PATH, classifier_activation: str='softmax'):
    model = get_model_arch(input_shape=input_shape, classes=classes, model_type=model_type, classifier_activation=classifier_activation)
    model = compile(model)
    model_path = get_model_ckpt_path(get_model_path(base_path=base_path, id=id))
    print(f"{model_path}")
    model.load_weights(model_path)
    return model

def load_ensemble(num_model: int, input_shape: Tuple[int, ...], classes: int, model_type: str, base_path: str=RESULT_PATH, classifier_activation: str='softmax'):
    model_list = []
    print(f"Loading Ensemble...")
    for i in range(num_model):
        model = load_model(id=i, input_shape=input_shape, classes=classes, model_type=model_type, 
                           base_path=base_path, classifier_activation=classifier_activation)
        model_list.append(model)
    
    return model_list

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

def train_an_ensemble(gpu_id: str, epoch: int, id: int=0, sel_label: List[int]=None, batch_size: int=32, width: int=512, base_path: str=None):
    exp_path = get_model_path(base_path=base_path, id=id)

    classifier_activation = 'softmax'
    is_onehot = True

    if sel_label is not None:
        classes = len(sel_label)
        if classes == 2:
            is_onehot = False
    else:
        classes = 10

    init_env(gpu_id=gpu_id)
    (x_train, y_train), (x_test, y_test) = get_CIFAR10(sel_label=sel_label, is_onehot=is_onehot)
    # print(f"y_train.shape OneHot: {y_train.shape}")
    model = get_model((32, 32, 3), classes, model_type=FINITE_CNN_MODEL, width=width, classifier_activation=classifier_activation)
    model, y_pred, loss, acc = train(model, x_train, y_train, x_test, y_test, batch_size, epoch, base_path=exp_path)

def plot_history(history):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def get_model_infos(model):
    layer_weights = []
    for layer in model.layers:
        w = layer.get_weights()
        print(f"size: {len(w)}| w: {w}")
        layer_weights.append(w)
# %%
if __name__ == '__main__':
    init_env('2')
    train_an_ensemble(gpu_id='2', width=2048, epoch=10, id=0, base_path=RESULT_PATH)
    # model = load_model(id=0, input_shape=(32, 32, 3), classes=10, model_type=FINITE_CNN_MODEL, base_path=RESULT_PATH, classifier_activation='softmax')
    # (x_train, y_train), (x_test, y_test) = get_CIFAR10(sel_label=None, is_onehot=True)
    # evaluate_model(model=model, x_test=x_test, y_test=y_test)

    # model_list = load_ensemble(num_model=50, input_shape=(32, 32, 3), classes=10, model_type=FINITE_CNN_MODEL, base_path=SHARE_DISK, classifier_activation='softmax')
    # model_list, loss, acc = evaluate_ensemble(model_list=model_list, x_test=x_test, y_test=y_test, eval_non_act=True)

    # get_model_infos(model_list[0])

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
