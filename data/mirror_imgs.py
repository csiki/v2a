import os, sys
import numpy as np
from glob import glob
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt


paths = sys.argv[1:]  # wildcards like path/to/images*.png

i = 0
for image_path in paths:
    # flip images vertically, l-r
    img = imread(image_path)
    img = np.flip(img, axis=1)

    flipped_img_path = image_path[:image_path.rfind('.')] + '_flip' + image_path[image_path.rfind('.'):]
    imsave(flipped_img_path, img)

    if (i + 1) % 100 == 0:
        print(i)
    i += 1
