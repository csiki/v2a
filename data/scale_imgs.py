import os
import tables
from glob import glob
import numpy as np
import tensorflow as tf
from skimage import img_as_float
from skimage.transform import resize
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt


paths = []
# paths.append('NUS-Hand-Posture-Dataset/BW/*.jpg')  # NUS
# paths.append('Sign-Language-Digits-Dataset/Dataset/*/*.jpg')  # sign language digits
# paths.append('Marcel-Test/*/uniform/*.ppm')  # Marcel
# paths.append('cambridge_subsample/*.jpg')

# paths.append('pl/original_images/*.jpg')
# paths.append('peipa/Peipa/*.pgm')

# paths.append('hand-synth/selected/*.jpg')

paths.append('own/table3/imgs/*.jpg')

# take img, resize to 160x120 if not that size
# possible sizes: 160x120, 100x100, 320x240
i = 0
path_to_save = 'own/table3/bw/'
for path in paths:
    for image_path in glob(path):
        img = imread(image_path)
        if img.shape[0] > 50:

            if 'original_images' in path:  # pl images are too large
                # portrait
                if img.shape[0] > img.shape[1]:
                    img = img[img.shape[0] // 6 : int(-img.shape[0] / 2.5),img.shape[1] // 6 : -img.shape[1] // 6]
                else:
                    img = img[img.shape[0] // 8 : -img.shape[0] // 8:, img.shape[1] // 6 : -img.shape[1] // 6]

            # format image
            img = img_as_float(img)
            if len(img.shape) == 3:  # RGB to grayscale
                img = img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114
            if img.shape[0] == img.shape[1]:
                img = resize(img, [160, 160])
                img = img[20:-20,:]  # cut 40 off vertically
                # plt.imshow(img)
                # plt.show()
            elif img.shape[0] == 240 and img.shape[1] == 320:
                img = resize(img, [120, 160], anti_aliasing=True)
            elif img.shape[0] == 120 and img.shape[1] == 160:
                pass
            else:
                img = resize(img, [120, 160], anti_aliasing=True)
                # print('para', img.shape)
                # continue

            # save image
            img_name = str(i) + '.png'
            imsave(path_to_save + img_name, img)

            if (i+1) % 500 == 0:
                print(i)

            i += 1

print(i, 'done')
