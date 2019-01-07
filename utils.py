import os
import scipy.misc
import numpy as np
from skimage import img_as_float
from skimage.transform import resize
import _pickle as cPickle
from tensorflow.python.client import device_lib


def get_image(image_path, crop_h, crop_w, is_crop=True, resize_h=64, resize_w=64, mode='RGB', normalize=True,
              complement=False, grayscale=False, v1_activation=False, only_layer=None, is_hdf5=True):
    if is_hdf5:
        img = image_path
        crop_fun = center_crop
        compl_fun = lambda x: np.uint8(255) - x
    elif v1_activation:
        img = get_v1_activation(image_path)
        crop_fun = center_crop_v1 if only_layer is None else center_crop
        compl_fun = lambda x: 1. - x
    else:
        img = scipy.misc.imread(image_path, mode=mode)
        crop_fun = center_crop
        compl_fun = lambda x: np.uint8(255) - x

    if only_layer is not None:
        img = img[:, :, only_layer]
    if is_crop:
        img = crop_fun(img, crop_h, crop_w, resize_h, resize_w)
    if complement:
        img = compl_fun(img)
    if normalize:
        img = img_as_float(img)
    if grayscale and only_layer is None:  # =flatten for v1_activation
        if len(img.shape) == 3:
            if v1_activation:
                img = np.max(img, axis=2)
            else:
                img = img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114

    return img


def get_v1_activation(image_path):  # TODO remove, depricated
    with open(image_path, 'rt') as f:
        h, w, d = (int(d) for d in f.readline().strip().split('\t'))
        v1_activation = np.zeros((h, w, d), dtype=np.float32)
        for line in f:
            r, c, o = (int(d) for d in line.split('\t'))
            v1_activation[r-1, c-1, o-1] = 1.  # -1 for stupid matlab indexing

    return v1_activation


def center_crop(x, crop_h, crop_w, resize_h, resize_w):  # FIXME depricated, only used for CelebA dataset
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    # return resize(x[j:j+crop_h, i:i+crop_w], (resize_h, resize_w))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_h, resize_w])  # kept this, cause it converts 0,255 to 0,1 not like img_as_float


def center_crop_v1(x, crop_h, crop_w, resize_h, resize_w):  # FIXME depricated, only used for CelebA dataset
    # does not perform actual image resize as center_crop
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return resize(x[j:j+crop_h, i:i+crop_w, :], (resize_h, resize_w, x.shape[2]),
                  anti_aliasing=True, preserve_range=True)


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
