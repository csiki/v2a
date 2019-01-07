from glob import glob
import tables, imageio
import numpy as np
from random import shuffle

# img_path = 'e:\\img_align_celeba_corf\\*.jpg'
# hdf5_path = 'e:\\celeba_corf2.hdf5'

# img_path = '/media/viktor/0C22201D22200DF0/img_align_celeba_corf/*.jpg'
# hdf5_path = '/media/viktor/0C22201D22200DF0/celeba_corf2.hdf5'

img_path = '/media/viktor/0C22201D22200DF0/hand_gestures/own/table3/v1/*.png'
hdf5_path = '/media/viktor/0C22201D22200DF0/hand_gestures/table3.hdf5'

# data_shape = (0, 218, 178)
data_shape = (0, 120, 160)
img_dtype = tables.UInt8Atom()
train_ratio = 0.9

imgs = [f for f in glob(img_path)]
shuffle(imgs)  # so train and test have random mixture of images

hdf5_file = tables.open_file(hdf5_path, mode='w')
train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)

# train
train_end = int(len(imgs) * train_ratio)
for i in range(train_end):
    if i % 1000 == 0 and i > 1:
        print(i)
    img = imageio.imread(imgs[i])
    # img = np.uint8(255) - imageio.imread(imgs[i]) # invert
    train_storage.append(img[None])

# test
for i in range(train_end, len(imgs)):
    if i % 1000 == 0:
        print(i)
    img = imageio.imread(imgs[i])
    # img = np.uint8(255) - imageio.imread(imgs[i]) # invert
    test_storage.append(img[None])

hdf5_file.close()
