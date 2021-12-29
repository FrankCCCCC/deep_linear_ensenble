from itertools import count
import os
from typing import List, Tuple, Union
from jax.core import Value
import json
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as onp
import glob
import jax.numpy as np
import cv2
import tensorflow_datasets as tfds
from jax import random, vmap
from scipy import stats

from util import pol2cart, check_make_dir

def shuffle_data(images, labels, seed=None):
    perm = onp.random.RandomState(seed).permutation(images.shape[0])
    images = images[perm]
    labels = labels[perm]
    return images, labels

# MODIFIED: Sample data for specific number of samples
def sample_data(images, sample_num, labels=None, seed=None):
    sample_range = images.shape[0] - 1
    if sample_num != -1:
        indices = onp.random.RandomState(seed).choice(sample_range, sample_num, replace=False)
        images = images[indices]
        
        if labels != None:
            labels = labels[indices]
            return images, labels
        return images
    else:
        if labels != None:
            return images, labels
        return images

def _partial_flatten_and_normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    x = onp.reshape(x, (x.shape[0], -1))
    return (x - onp.mean(x)) / onp.std(x)

def _flatten(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return onp.reshape(x, (x.shape[0], -1))/255

def _normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return x / 255


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return onp.array(x[:, None] == onp.arange(k), dtype)

def rgb2gray(rgb):
    return onp.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_labels(train_size: int, noise_size: int):
    labels = onp.concatenate([
        onp.ones((train_size,), onp.float64), 
        onp.zeros((noise_size,), onp.float64)
    ])
    labels = _one_hot(labels, 2, dtype=np.float64)

    target_label = onp.ones((train_size + noise_size,), onp.float64)
    target_label = _one_hot(target_label, 2, dtype=np.float64)

    return labels, target_label

class ImageNet():
    TRAINING_DATA = 'train'
    TESTING_DATA = 'test'
    VALIDATING_DATA = 'val'
    MID_PATH = 'ILSVRC/Data/CLS-LOC'
    CLASS_INDEX = 'imagenet_class_index.json'

    def __init__(self, data_path: str):
        self.data_path = data_path

    def __get_class_dir(self, classes: Tuple[List[str], str]=None):
        with open(os.path.join(self.data_path, self.CLASS_INDEX)) as json_file:
            data = json.load(json_file)

        class_dict = {}
        for k in data.keys():
            folder, class_name = data[k][0], data[k][1]
            class_dict[class_name] = [k, folder]

        if isinstance(classes, list):
            return [class_dict.get(c, [None, None])[1] for c in classes]
        elif isinstance(classes, str):
            return [class_dict.get(classes, None)]
        elif classes == None:
            return [class_dict.get(k, [None, None])[1] for k in class_dict.keys()]
        else:
            raise TypeError(f"Argument classes should be a string, List of string or None, but not {type(classes)}")

    def __get_type_path(self, types: Tuple[List[str], str]=None):
        if isinstance(types, list):
            return [os.path.join(self.data_path, t) for t in types]
        elif isinstance(types, str):
            return [os.path.join(self.data_path, types)]
        else:
            raise TypeError(f"Argument classes should be a string, List of string or None, but not {type(types)}")

    def __get_img_path(self, dir_paths: Tuple[List[str], str]):
        img_list = []
        if isinstance(dir_paths, list):
            for d in dir_paths:
                for (dirpath, dirnames, filenames) in os.walk(d):
                    img_list.append(os.path.join(d, filenames))
            return img_list
        elif isinstance(dir_paths, str):
            for (dirpath, dirnames, filenames) in os.walk(dir_paths):
                    img_list.append(os.path.join(dir_paths, filenames))
            return img_list
        else:
            raise TypeError(f"Argument classes should be a string, List of string or None, but not {type(types)}")

    def __read_img(self, path: str):
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def __read_images(self, img_path_list: List[str], n_jobs: int=-1):
        images = Parallel(n_jobs=n_jobs)(
            delayed(self.__read_img)(f) for f in img_path_list
        )
        return np.stack(images, axis=0)
    
    def __get_imgs(self, data_paths: List[str], class_folders: List[str]):
        img_path_list = []
        for d in data_paths:
            for c in class_folders:
                img_path_list += self.__get_img_path(os.path.join(d, c))
        images = self.__read_images(img_path_list)
        return images

    def get_data(self, types: List[str]=None, classes: List[str]=None):
        class_folders = self.__get_class_dir(classes=classes)
        data_paths = self.__get_type_path(types=types)
        images = self.__get_imgs(data_paths=data_paths, class_folders=class_folders)
        return images

class Dataset():
    MNIST = 'mnist'
    CELEB_A = 'celeb_a'
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    IMAGENET = 'imagenet'
    GAUSSIAN_8 = 'gaussian_8'
    GAUSSIAN_25 = 'gaussian_25'

    MNIST_SHAPE = (28, 28, 1)
    CIFAR10_SHAPE = (32, 32, 3)
    CELEB_A_SHAPE = (64, 64, 3)
    GAUSSIAN_8_SHAPE = (2, )
    GAUSSIAN_25_SHAPE = (2, )

    CELEB_A_ANNO_DIR = 'Anno'
    CELEB_A_IMG64_DIR = 'img_align_celeba_64'
    CELEB_A_ATTRS_FILE = 'list_attr_celeba.txt'

    def __init__(self, dataset_name: str, seed: int, celeb_a_path: str=None, flatten: bool=False) -> None:
        self.dataset_name = dataset_name
        self.seed = seed
        self.celeb_a_path = celeb_a_path
        self.flatten = flatten
        self.ng = NoiseGenerator(random_seed=seed)

        self.image_shape, self.vec_size = Dataset.get_dataset_shape(dataset_name=dataset_name)
    
    def __get_dataset(self, n_train: int=None, n_test: int=None, permute_train: bool=False, 
                      normalize: bool=False):
        """Download, parse and process a dataset to unit scale and one-hot labels."""

        ds_builder = tfds.builder(self.dataset_name)
        ds_train, ds_test = tfds.as_numpy(
            tfds.load(
                self.dataset_name + ':3.*.*',
                split=['train' + ('[:%d]' % n_train if n_train is not None else ''),
                    'test' + ('[:%d]' % n_test if n_test is not None else '')],
                batch_size=-1,
                as_dataset_kwargs={'shuffle_files': False}))

        train_images, train_labels, test_images, test_labels = (ds_train['image'],
                                                                ds_train['label'],
                                                                ds_test['image'],
                                                                ds_test['label'])
        num_classes = ds_builder.info.features['label'].num_classes
        
        if self.flatten and normalize:
            train_images = _partial_flatten_and_normalize(train_images)
            test_images = _partial_flatten_and_normalize(test_images)
        elif self.flatten:
            train_images = _flatten(train_images)
            test_images = _flatten(test_images)
        else:
            train_images = _normalize(train_images)
            test_images = _normalize(test_images)
            
        train_labels = _one_hot(train_labels, num_classes)
        test_labels = _one_hot(test_labels, num_classes)

        if permute_train:
            perm = onp.random.RandomState(0).permutation(train_images.shape[0])
            train_images = train_images[perm]
            train_labels = train_labels[perm]

        return train_images, train_labels, test_images, test_labels

    @staticmethod
    def get_dataset_shape(dataset_name: str):
        if dataset_name == Dataset.MNIST:
            image_shape = Dataset.MNIST_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.CIFAR10:
            image_shape = Dataset.CIFAR10_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.CELEB_A:
            image_shape = Dataset.CELEB_A_SHAPE
            vec_size = image_shape[0] * image_shape[1] * image_shape[2]
        elif dataset_name == Dataset.GAUSSIAN_8:
            image_shape = Dataset.GAUSSIAN_8_SHAPE
            vec_size = image_shape[0]
        elif dataset_name == Dataset.GAUSSIAN_25:
            image_shape = Dataset.GAUSSIAN_25_SHAPE
            vec_size = image_shape[0]
        else:
            raise ValueError(f"No such dataset named {dataset_name}")
        return image_shape, vec_size

    def get_data_shape(self):
        return self.image_shape, self.vec_size

    def reset_seed(self, seed: int):
        self.seed = seed
        self.ng = NoiseGenerator(random_seed=seed)

    def set_sample_size(self, noise_size: int=None, train_size: int=None, dataset_size: int=None) -> 'Dataset':
        """
        If train_size is None(default), switch to GD. Otherwise, use SGD with batch size = train_size
        """
        self.noise_size = noise_size
        self.dataset_size = dataset_size

        if train_size is None:
            self.train_size = dataset_size
        if isinstance(train_size, str):
            if train_size.lower() == 'none':
                self.train_size = dataset_size
        if isinstance(train_size, int):
            self.train_size = train_size

        return self

    def gen_noise(self):
        if self.noise_size is not None:
            x_noise = self.ng.gen_noise(noise_size=self.noise_size, image_shape=self.image_shape, flatten=self.flatten)
            return x_noise
        else:
            raise ValueError(f"Please call method set_sample_size and set noise_size first")

    def gen_labels(self):
        if (self.train_size is not None) and (self.noise_size is not None):
            return get_labels(self.train_size, self.noise_size)
        else:
            raise ValueError(f"Please call method set_sample_size and set train_size, noise_size first")

    def gen_data_attrs(self, target_class: int=None, attrs: List[str]=None) -> Tuple[np.array, np.array]:
        # read data
        if self.dataset_name == self.MNIST or self.dataset_name == self.CIFAR10:
            x_train_all, y_train_all, x_test_all, y_test_all = tuple(
                np.array(x) for x in self.__get_dataset(None, None)
            )

            _target_class = None
            if isinstance(target_class, str):
                if target_class.lower() == 'none':
                    _target_class = None
                else:
                    _target_class = int(target_class)
            elif isinstance(target_class, float) or isinstance(target_class, int):
                _target_class = int(target_class)
            else:
                if target_class is None:
                    _target_class = None

            if _target_class is None:
                # shuffle
                x_train_all, y_train_all = shuffle_data(x_train_all, y_train_all, self.seed)
            else:
                # get target class images
                x_train_all = x_train_all[np.argmax(y_train_all, axis=1)==_target_class]
                y_train_all = y_train_all[np.argmax(y_train_all, axis=1)==_target_class]

            x_train_all, y_train_all = sample_data(images=x_train_all, labels=y_train_all, sample_num=self.dataset_size, seed=self.seed)
            x_train = x_train_all[:self.train_size]
            y_train = y_train_all[:self.train_size]
                
        elif self.dataset_name == self.CELEB_A:
            if self.celeb_a_path is None:
                raise BaseException("Please specify the path of CELEB_A dataset")

            # parse attribute file
            file_list = []
            attr_list_file_path = os.path.join(self.celeb_a_path, self.CELEB_A_ANNO_DIR, self.CELEB_A_ATTRS_FILE)
            f = open(attr_list_file_path, 'r')
            # print(f"Anno - {self.CELEB_A_ATTRS_FILE}: {attr_list_file_path}")
            file_list = f.readlines()
            f.close()
            attr_to_num = {file_list[1].split(' ')[:-1][idx]: idx for idx in range(len(file_list[1].split(' ')[:-1]))}
            file_name_len = 10
            
            # select images with attributes
            x_train_all = []
            for i, line in enumerate(file_list[2:]):
                # attriutes
                cond_list = [True]
                if (attrs != None) and (attrs != []):
                    for attr in attrs:
                        offset_1 = file_name_len+3*(attr_to_num[attr])
                        offset_2 = file_name_len+3*(attr_to_num[attr]+1)
                        
                        cond_n = (int(line[offset_1:offset_2]) == 1)
                        cond_list.append(cond_n)
                
                if np.all(np.array(cond_list)):
                    file_path = os.path.join(self.celeb_a_path, self.CELEB_A_IMG64_DIR, line[:10])
                    # print(f"file_path: {file_path}")
                    # cv2.imread may not work fine due to bugs
                    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    # image = cv2.cv.LoadImage(file_path, cv2.CV_LOAD_IMAGE_COLOR)
                    x_train_all.append(image)
                    
                    # print(f"{i}-th Image Shape: {type(image)} {image.shape}: {image}")
            
            # to numpy array
            x_train_all = onp.stack(x_train_all, axis=0)
            # print(f"x_train_all after stack: {x_train_all.shape}")
            x_train_all = x_train_all.astype(np.float64)
            x_train_all /= 255
            # MODIFIED: Sample data
            if self.flatten:
                x_train_all = np.reshape(x_train_all, (x_train_all.shape[0], -1))
                # print(f"x_train_all after flatten: {x_train_all.shape}")
            x_train_all = sample_data(images=x_train_all, sample_num=self.dataset_size, seed=self.seed)
            # print(f"x_train_all after sample: {x_train_all.shape}")
            x_train = x_train_all[:self.train_size]
            # print(f"x_train generate: {x_train.shape}")
        
        elif self.dataset_name == self.GAUSSIAN_8:
            width = 1
            height = 1
            r = 0.4
            n_mode = 8
            scale = 0.002
            gauss_gen = Mixed_Gaussian()
            x_train_all = gauss_gen.circle(n_sample=self.dataset_size, n_mode=n_mode, r=r, scale=scale, width=width, height=height)
            x_train = x_train_all[:self.train_size]

        elif self.dataset_name == self.GAUSSIAN_25:
            width = 1
            height = 1
            sqrt_mode = 5
            scale = 0.0005
            gauss_gen = Mixed_Gaussian()
            x_train_all = gauss_gen.square(n_sample=self.dataset_size, sqrt_mode=sqrt_mode, scale=scale, width=width, height=height)
            x_train = x_train_all[:self.train_size]
            
        return np.array(x_train), np.array(x_train_all)
class Mixed_Gaussian():
    CIRCLE_GEN_TYPE = 'CIRCLE'
    SQUARE_GEN_TYPE = 'SQUARE'
    HEATMAP = 'heatmap'
    SCATTER = 'scatter'

    def __init__(self, seed: int=0):
        self.rnd_key = random.PRNGKey(seed)
    
    def __get_key(self) -> np.ndarray:
        key, subkey = random.split(self.rnd_key)
        self.rnd_key = subkey
        return subkey

    def __circle_gauss(self, n_mode: int, r: float, scale: float, width: float, height: float):
        mean_vecs = []
        cov_mats = []
        for i in range(n_mode):
            theta = 2 * np.pi / n_mode * i
            x, y = pol2cart(r, theta)
            mean_vec = np.array([x + width/2, y + height/2])
            cov_mat = np.eye(2) * scale
            mean_vecs.append(mean_vec)
            cov_mats.append(cov_mat)

        return np.array(mean_vecs), np.array(cov_mats)

    def __square_gauss(self, sqrt_mode: int, scale: float, width: float, height: float):
        mean_vecs = []
        cov_mats = []
        x_start = width * 0.15
        y_start = height * 0.15
        x_gap = (width - 2 * x_start) / sqrt_mode
        y_gap = (height - 2 * y_start) / sqrt_mode
        # print(f"X_start: {x_start} Y_start: {y_start} X_gap: {x_gap} Y_gap: {y_gap}")
        for i in range(sqrt_mode):
            for j in range(sqrt_mode):
                mean_vec = np.array([x_start + x_gap * i, y_start + y_gap * j ])
                cov_mat = np.eye(2) * scale
                mean_vecs.append(mean_vec)
                cov_mats.append(cov_mat)

        return np.array(mean_vecs), np.array(cov_mats)
    
    def __mode_count(self, n_sample: int, n_mode: int):
        mode_choices = random.randint(key=self.__get_key(), shape=(n_sample,), minval=0, maxval=n_mode)
        uniques, counts = np.unique(mode_choices, return_counts=True)
        return counts

    def __sample_mixed_gauss(self, n_mode: int, mean_vecs: np.ndarray, cov_mats: np.ndarray, counts: np.array):
        samples = []
        for i in range(n_mode):
            xy = random.multivariate_normal(key=self.__get_key(), mean=mean_vecs[i], cov=cov_mats[i], shape=(counts[i], ))
            samples.append(xy)
        samples = np.concatenate(samples, axis=0)
        return random.permutation(key=self.__get_key(), x=samples)

    def circle(self, n_sample: int, n_mode: int, r: Union[int, float], scale: Union[int, float], width: float, height: float) -> np.ndarray:
        counts = self.__mode_count(n_sample=n_sample, n_mode=n_mode)
        mean_vecs, cov_mats = self.__circle_gauss(n_mode=n_mode, r=r, scale=scale, width=width, height=height)

        return self.__sample_mixed_gauss(n_mode=n_mode, mean_vecs=mean_vecs, cov_mats=cov_mats, counts=counts)

    def square(self, n_sample: int, sqrt_mode: int, scale: Union[int, float], width: float, height: float) -> np.ndarray:
        n_mode = sqrt_mode * sqrt_mode
        counts = self.__mode_count(n_sample=n_sample, n_mode=n_mode)
        mean_vecs, cov_mats = self.__square_gauss(sqrt_mode=sqrt_mode, scale=scale, width=width, height=height)

        return self.__sample_mixed_gauss(n_mode=n_mode, mean_vecs=mean_vecs, cov_mats=cov_mats, counts=counts)

    @staticmethod
    def visualize(datas: np.ndarray, width: int, height: int, title=None, plot_type: str=None, is_axis: bool=True, fig_path: str=None, is_show_fig: bool=False, dpi=300, fig_size: Tuple=(9, 9), background_color: str=None):
        if fig_path is not None:
            fig_dir = os.path.dirname(fig_path)
            check_make_dir(fig_dir)

        plt.clf()
        fig = plt.figure(figsize=fig_size, dpi=dpi, facecolor=background_color)
        if not is_axis:
            plt.axis('off')

        if title is not None:
            plt.title(title)
        plt.tight_layout()

        if plot_type is Mixed_Gaussian.SCATTER:
            xs = datas[:, 0]
            ys = datas[:, 1]
            plt.scatter(xs, ys)
        else:
            xs = datas[:, 0]
            ys = datas[:, 1]
            heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=(width, height))
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.imshow(heatmap.T, extent=extent, origin='lower')

        if fig_path is not None:
            plt.savefig(fig_path)

        if is_show_fig:
            plt.show()
        plt.close()

def n_mode_gaussian(n_sample: int, n_mode: int, r: Union[int, float], scale: Union[int, float], width: int, height: int):
    gaussians = []
    for i in range(n_mode):
        idt_mat = np.eye(2) * scale / 5
        theta = 2 * np.pi / n_mode * i
        x, y = pol2cart(r, theta)
        gauss = stats.multivariate_normal((x + width/2, y + height/2), cov=idt_mat)
        gaussians.append(gauss)
    
    mode_choices = np.random.choice(n_mode, n_sample)
    samples = np.array([gaussians[choice].rvs(size=1) for choice in mode_choices])
    return samples

class NoiseGenerator():
    def __init__(self, random_seed: int):
        self.rnd_key = random.PRNGKey(random_seed)

    def __get_key(self) -> np.ndarray:
        key, subkey = random.split(self.rnd_key)
        self.rnd_key = subkey
        return subkey

    def gen_noise(self, noise_size: int, image_shape: Tuple, flatten: bool=False):
        #Generate noise
        x_noise = random.uniform(self.__get_key(), shape=(noise_size, *image_shape), minval=0, maxval=1.0)

        if flatten:
            x_noise = np.reshape(x_noise, (noise_size, -1))
        return x_noise

class DiffAugment():
    def __init__(self, random_seed: int, policy: str='', channels_first: bool=False):
        self.policy = policy
        self.channels_first = channels_first
        self.rnd_key = random.PRNGKey(random_seed)

    def __get_key(self) -> np.ndarray:
        key, subkey = random.split(self.rnd_key)
        self.rnd_key = subkey
        return subkey

    def augment(self, x):
        AUGMENT_FNS = {
            'color': [self.rand_brightness, self.rand_saturation, self.rand_contrast],
            'translation': [self.rand_translation],
            'cutout': [self.rand_cutout],
            'none': []
        }

        if self.policy is not None:
            if self.channels_first:
                x = np.transpose(x, (0, 2, 3, 1))
            for p in self.policy.split(','):
                for f in AUGMENT_FNS[p]:
                    x = f(x)
            if self.channels_first:
                x = np.transpose(x, (0, 3, 1, 2))
        return x


    def rand_brightness(self, x):
        magnitude = random.uniform(key=self.__get_key(), shape=[np.shape(x)[0], 1, 1, 1], minval=0, maxval=1, dtype=np.float32) - 0.5
        x = x + magnitude
        return x

    def rand_saturation(self, x):
        magnitude = random.uniform(key=self.__get_key(), shape=[np.shape(x)[0], 1, 1, 1], minval=0, maxval=1, dtype=np.float32) * 2
        x_mean = np.mean(x, axis=3, keepdims=True)
        x = (x - x_mean) * magnitude + x_mean
        return x

    def rand_contrast(self, x):
        magnitude = random.uniform(key=self.__get_key(), shape=[np.shape(x)[0], 1, 1, 1], minval=0, maxval=1, dtype=np.float32) + 0.5
        x_mean = np.mean(x, axis=[1, 2, 3], keepdims=True)
        x = (x - x_mean) * magnitude + x_mean
        return x

    def __gather_nd_unbatched(self, params, indices):
        return params[tuple(np.moveaxis(indices, -1, 0))]

    def __gather_nd(self, params, indices, batch=False):
        if not batch:
            return self.__gather_nd_unbatched(params, indices)
        else:
            return vmap(self.__gather_nd_unbatched, (0, 0), 0)(params, indices)

    def __scatter_nd(self, indices, updates, shape):
        zeros = np.zeros(shape, updates.dtype)
        key = tuple(np.moveaxis(indices, -1, 0))
        return zeros.at[key].add(updates)

    def rand_translation(self, x, ratio=0.125):
        batch_size = np.shape(x)[0]
        image_size = np.shape(x)[1:3]
        # shift = np.intc(np.float32((image_size) * ratio + 0.5))
        shift = np.asarray((np.asarray(image_size, dtype=np.float32) * ratio + 0.5), dtype=np.int32)
        translation_x = np.asarray(random.uniform(key=self.__get_key(), shape=[batch_size, 1], minval=-shift[0], maxval=shift[0] + 1), dtype=np.int32)
        translation_y = np.asarray(random.uniform(key=self.__get_key(), shape=[batch_size, 1], minval=-shift[1], maxval=shift[1] + 1), dtype=np.int32)
        grid_x = np.clip(np.expand_dims(np.arange(image_size[0], dtype=np.int32), 0) + translation_x + 1, 0, image_size[0] + 1)
        grid_y = np.clip(np.expand_dims(np.arange(image_size[1], dtype=np.int32), 0) + translation_y + 1, 0, image_size[1] + 1)
        # gather_nd in Jax: https://github.com/google/jax/discussions/6119
        x = self.__gather_nd(np.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), np.expand_dims(grid_x, -1), batch=True)
        x = np.transpose(self.__gather_nd(np.pad(np.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]), np.expand_dims(grid_y, -1), batch=True), [0, 2, 1, 3])
        return x

    def rand_cutout(self, x, ratio=0.5):
        batch_size = np.shape(x)[0]
        image_size = np.shape(x)[1:3]
        # cutout_size = np.intc(np.float32(image_size) * ratio + 0.5)
        cutout_size = np.asarray((np.asarray(image_size, dtype=np.float32) * ratio + 0.5), dtype=np.int32)
        offset_x = np.asarray(random.uniform(key=self.__get_key(), shape=[np.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2)), dtype=np.int32)
        offset_y = np.asarray(random.uniform(key=self.__get_key(), shape=[np.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2)), dtype=np.int32)
        grid_batch, grid_x, grid_y = np.meshgrid(np.arange(batch_size, dtype=np.int32), np.arange(cutout_size[0], dtype=np.int32), np.arange(cutout_size[1], dtype=np.int32), indexing='ij')
        cutout_grid = np.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
        mask_shape = np.stack([batch_size, image_size[0], image_size[1]])
        cutout_grid = np.maximum(cutout_grid, 0)
        cutout_grid = np.minimum(cutout_grid, np.reshape(mask_shape - 1, [1, 1, 1, 3]))
        # scatter_nd in Jax: https://github.com/google/jax/discussions/3658
        mask = np.maximum(1 - self.__scatter_nd(cutout_grid, np.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=np.float32), mask_shape), 0)
        x = x * np.expand_dims(mask, axis=3)
        return x