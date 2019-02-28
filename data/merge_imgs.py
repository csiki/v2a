import tables, imageio
from random import shuffle
import sys


imgs = sys.argv[1:-1]  # like path/to/contour/imgs*.png
hdf5_path = sys.argv[-1]  # like output/path/something.hdf5

data_shape = (0, 120, 160)
img_dtype = tables.UInt8Atom()
train_ratio = 0.9

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
    train_storage.append(img[None])

# test
for i in range(train_end, len(imgs)):
    if i % 1000 == 0:
        print(i)
    img = imageio.imread(imgs[i])
    test_storage.append(img[None])

hdf5_file.close()
