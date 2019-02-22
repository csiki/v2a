import os
import scipy.misc
import numpy as np
from skimage import img_as_float
from skimage.transform import resize
import _pickle as cPickle
from tensorflow.python.client import device_lib
import tensorflow as tf


def dense(x, input_features, output_features, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_features, output_features], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_features], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def get_image(image_path, grayscale=False):
    img = image_path
    img = img_as_float(img)

    if grayscale and len(img.shape) == 3 and img.shape[2] == 3:  # colored to grayscale
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

    return img


def merge_color(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if len(image.shape) == 2:  # grayscale, probably v1 activation
            image = image
            img[j * h:j * h + h, i * w:i * w + w, 0] = image
            img[j * h:j * h + h, i * w:i * w + w, 1] = image
            img[j * h:j * h + h, i * w:i * w + w, 2] = image
        elif image.shape[2] != 3:  # v1 activation
            image = np.max(image, axis=2)
            img[j * h:j * h + h, i * w:i * w + w, 0] = image
            img[j * h:j * h + h, i * w:i * w + w, 1] = image
            img[j * h:j * h + h, i * w:i * w + w, 2] = image
        else:
            img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def ims(name, img, cmin=0, cmax=1):
    # print(img[:10][:10])
    if not os.path.exists(os.path.dirname(name)):
        os.mkdir(os.path.dirname(name))
    scipy.misc.toimage(img, cmin=cmin, cmax=cmax).save(name)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
