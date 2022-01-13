# %%
import os
from turtle import st
from typing import List, Tuple, Union
import pickle
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, backend
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import update
from tensorflow.python.types.core import Value
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# from dataset import Dataset
from scalablerunner.taskrunner import TaskRunner
from model import ModelMgr

def init_env(gpu_id: str, float64: bool=False):
    if float64:
        tf.keras.backend.set_floatx('float64')
    else:
        tf.keras.backend.set_floatx('float32')
    if gpu_id is not None:
        if isinstance(gpu_id, str):
            if gpu_id.lower() != 'none':
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                return
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
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

class RecordMgr():
    CKPT_PATH = 'checkpoints'
    RESULT_PATH = 'results'
    CKPT_ENSEMBLE_PREFIX = 'ensemble_'
    CKPT_NAME = 'ckpt'
    SAVED_MOEDL_NAME = 'saved_model.h5'
    SAVED_LIN_REG_NAME = 'saved_lin_reg.pkl'
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

    def get_saved_lin_reg_path(self):
        saved_model_path = os.path.join(self.get_model_ckpt_path(), RecordMgr.SAVED_LIN_REG_NAME)
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

    def save_lin_reg(self, w):
        save_pkl(obj=w, file=self.get_saved_lin_reg_path())

    def load_lin_reg(self):
        return load_pkl(file=self.get_saved_lin_reg_path())

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

class Training():
    def __init__(self, batch_size: int=32, epoch: int=30, lr: float=0.001, l2_regular: float=1e-2, normalize: bool=False, base_path: str=None):
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.base_path = base_path
        self.l2_regular = l2_regular
        self.normalize = normalize
        self.scaler = None
        if normalize:
            self.scaler = StandardScaler()

    def __init_training(self, model: tf.keras.Model, base_path: str=None, config: dict=None):
        model = ModelMgr.compile(model, learning_rate=self.lr)
        record_mgr = RecordMgr(base_path=base_path)
        visulize_mgr = VisulaizeMgr(base_path=base_path)

        if config is not None:
            record_mgr.save_config(config=config)
        return model, record_mgr, visulize_mgr

    @staticmethod
    def __svd_inverse(M: Union[np.ndarray, tf.Tensor]):
        s, u, v = tf.linalg.svd(M)
        inv_diag = 1/s * tf.eye(s.shape[0])
        # print(f"v: {v.shape}, inv_diag: {inv_diag.shape}, u: {u.shape}")
        return tf.linalg.matmul(tf.linalg.matmul(v, inv_diag), tf.transpose(u))

    @staticmethod
    def __inverse(M: Union[np.ndarray, tf.Tensor]):
        return tf.linalg.inv(M)

    @staticmethod
    def __solve_lin_sys(A: Union[np.ndarray, tf.Tensor], y: Union[np.ndarray, tf.Tensor], l2_regular: float):
        cov = tf.linalg.matmul(tf.transpose(A), A)
        pre_inv = cov + l2_regular * tf.eye(cov.shape[0])
        trans_label = tf.linalg.matmul(tf.transpose(A), y)
        return tf.linalg.matmul(Training.__svd_inverse(M=pre_inv), trans_label)

    @staticmethod
    def __compute_w(model: tf.keras.Model, x_train, y_train, l2_regular: float, normalize: bool):
        print("Dataset is embedding...")
        start = datetime.now()
        x_train_embed = Training.model_predict(model=model, x=x_train)
        end = datetime.now()
        print(f"Dataset has been Embedded - Time: {end - start}")
        scaler = None
        if normalize:
            scaler = StandardScaler()
            x_train_embed = scaler.fit_transform(X=x_train_embed)
        return Training.__solve_lin_sys(A=x_train_embed, y=y_train, l2_regular=l2_regular), scaler

    @staticmethod
    def MSE_CatAcc(y_pred, y_test):
        loss_metric = tf.keras.losses.MeanSquaredError()
        eval_metric = tf.keras.metrics.CategoricalAccuracy()

        eval_metric.update_state(y_test, y_pred)
        pred_acc = eval_metric.result().numpy()
        pred_loss = loss_metric(y_test, y_pred)
        return pred_loss, pred_acc

    @staticmethod
    def model_predict(model: tf.keras.Model, x: Union[np.ndarray, tf.Tensor]):
        y_pred = model.predict(x)
        return y_pred

    @staticmethod
    def lin_reg_predict(model: tf.keras.Model, w, x_test: Union[np.ndarray, tf.Tensor], scaler):
        x_test_embed = model.predict(x_test)
        if scaler is not None:
            x_test_embed = scaler.transform(x_test_embed)
        y_pred = tf.linalg.matmul(x_test_embed, w)
        return y_pred

    @staticmethod
    def evaluate_linear_reg(model: tf.keras.Model, w, x_test, y_test, scaler):
        y_pred = Training.lin_reg_predict(model=model, w=w, x_test=x_test, scaler=scaler)
        loss, acc = Training.MSE_CatAcc(y_pred=y_pred, y_test=y_test)
        return model, loss, acc

    @staticmethod
    def evaluate_model(model: tf.keras.Model, x_test, y_test):
        # Built-in method
        # loss, acc = model.evaluate(x_test, y_test, verbose=2)

        # For testing new method
        y_pred = Training.model_predict(model=model, x=x_test)
        loss, acc = Training.MSE_CatAcc(y_pred=y_pred, y_test=y_test)
        return model, loss, acc

    def linear_reg_fit(self, model: tf.keras.Model, x_train, y_train, x_test, y_test, base_path: str=None, config: dict=None):
        model, record_mgr, visulize_mgr = self.__init_training(model=model, base_path=base_path, config=config)

        print("Solving linear system...")
        start = datetime.now()
        w, scaler = Training.__compute_w(model=model, x_train=x_train, y_train=y_train, l2_regular=self.l2_regular, normalize=self.normalize)
        end = datetime.now()
        print(f"Linear system has been sovled - Time: {end - start}")
        self.scaler = scaler

        if base_path is not None:
            record_mgr.save_model(model=model)
            record_mgr.save_lin_reg(w=w)

        model, loss, acc = self.evaluate_linear_reg(model=model, w=w, x_test=x_test, y_test=y_test, scaler=self.scaler)
        return model, loss, acc

    def keras_fit(self, model: tf.keras.Model, x_train, y_train, x_test, y_test, base_path: str=None, config: dict=None):
        model, record_mgr, visulize_mgr = self.__init_training(model=model, base_path=base_path, config=config)

        if base_path is not None:
            ckpt_callback = ModelMgr.get_ckpt_callback(checkpoint_path=record_mgr.get_saved_model_path())
            history = model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epoch, verbose="auto",
                    validation_data=(x_test, y_test), validation_freq=1, callbacks=[ckpt_callback])
            record_mgr.save_history(history=history.history)
        else:
            # No checkpoint
            history = model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epoch, verbose="auto",
                    validation_data=(x_test, y_test), validation_freq=1)

        visulize_mgr.plot_history(history=record_mgr.load_history())

        model, loss, acc = self.evaluate_model(model=model, x_test=x_test, y_test=y_test)
        return model, loss, acc

    def train_a_model(self, gpu_id: str, id: int=0, sel_label: List[int]=None, layer_num: int=5, width: int=512, kernel_size: Tuple[int, int]=(3, 3), conv_block: int=1,
                      model_type: str=ModelMgr.FINITE_CNN_MODEL, classifier_activation: str='softmax', is_freeze: bool=True, closed_form: bool=False, float64: bool=False):
        is_onehot = True
        config = locals()
        del config['self']
        print(config)

        if sel_label is not None:
            classes = len(sel_label)
            if classes == 2:
                is_onehot = False
        else:
            classes = 10

        if id is not None:
            # Ensemble directory
            exp_path = RecordMgr.get_ensemble_model_path(base_path=self.base_path, id=id)
        else:
            # Hyperparameter naming
            exp_path = os.path.join(self.base_path, f'{model_type}_ln-{layer_num}_w-{width}_cb-{conv_block}_act-{classifier_activation}_ep-{self.epoch}_bs-{self.batch_size}_lr-{self.lr}')
        model_mgr = ModelMgr(input_shape=(32, 32, 3), classes=classes, model_type=model_type, layer_num=layer_num, width=width, kernel_size=kernel_size, conv_block=conv_block, classifier_activation=classifier_activation, is_freeze=is_freeze)

        init_env(gpu_id=gpu_id, float64=float64)
        (x_train, y_train), (x_test, y_test) = get_CIFAR10(sel_label=sel_label, is_onehot=is_onehot)
        # print(f"y_train.shape OneHot: {y_train.shape}")
        model = model_mgr.get_model()
        if closed_form:
            model, loss, acc = self.linear_reg_fit(model, x_train, y_train, x_test, y_test, base_path=exp_path, config=config)
        else:
            model, loss, acc = self.keras_fit(model, x_train, y_train, x_test, y_test, base_path=exp_path, config=config)

        print(f"Evaluate - Acc: {acc} | Loss: {loss}")

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
        y_pred = model_predict(model=model, x_test=x_test)
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
    print(f"y_pred_mean.shape: {y_pred_mean.shape}")
    eval_metric.update_state(y_test, y_pred_mean)
    acc = eval_metric.result().numpy()
    loss = loss_metric(y_test, y_pred_mean)

    print(f"{len(model_list)} Ensemble Eval - Acc: {acc} | Loss: {loss}")
    
    return model_list, loss, acc

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
    # init_env('1', float64=True)
    training_loop = Training(batch_size=32, epoch=30, l2_regular=1e-2, normalize=True, base_path=RecordMgr.RESULT_PATH)
    # train_a_model(gpu_id='0', layer_num=5, width=1024, conv_block=1, epoch=30, id=0, model_type=ModelMgr.FINITE_CNN_MODEL, classifier_activation=None, is_freeze=True, base_path=RecordMgr.RESULT_PATH)
    training_loop.train_a_model(gpu_id='1', layer_num=5, width=2048, kernel_size=(3, 3), conv_block=1, id=0, model_type=ModelMgr.FINITE_CNN_MODEL, classifier_activation=None, is_freeze=True, 
                                closed_form=True, float64=False)

    # record_mgr = RecordMgr.get_ensemble_model_RecordMgr(base_path=RecordMgr.RESULT_PATH, id=0)
    # model = record_mgr.load_model()
    # (x_train, y_train), (x_test, y_test) = get_CIFAR10(sel_label=None, is_onehot=True)
    # evaluate_model(model=model, x_test=x_test, y_test=y_test)

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
