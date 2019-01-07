import os
import numpy as np
from glob import glob
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt


paths = []
paths.append('mergedown_v1/*.png')

i = 0
for path in paths:
    for image_path in glob(path):

        # flip images vertically, l-r
        img = imread(image_path)
        img = np.flip(img, axis=1)
        plt.imshow(img)
        plt.show()
        
        flipped_img_path = image_path[:image_path.rfind('.')] + '_flip' + image_path[image_path.rfind('.'):]
        # imsave(flipped_img_path, img) FIXME

        if (i+1) % 100 == 0:
            print(i)
        i += 1
