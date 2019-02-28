import sys
from skimage import img_as_float
from skimage.transform import resize
from scipy.misc import imread, imsave


images_path = sys.argv[1:-1]  # wildcards like path/to/images*.png
path_to_save = sys.argv[-1]  # like path/to/output/

# take img, resize it to 160x120
# possible sizes: 160x120, 100x100, 320x240
i = 0
for image_path in images_path:
    img = imread(image_path)
    if img.shape[0] > 50:

        # format image
        img = img_as_float(img)
        if len(img.shape) == 3:  # RGB to grayscale
            img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        # resize
        if img.shape[0] == img.shape[1]:
            img = resize(img, [160, 160])
            img = img[20:-20, :]  # cut 40 off vertically, ratio intact
        elif img.shape[0] == 240 and img.shape[1] == 320:
            img = resize(img, [120, 160], anti_aliasing=True)
        elif img.shape[0] == 120 and img.shape[1] == 160:
            pass
        else:
            img = resize(img, [120, 160], anti_aliasing=True)

        # save image
        img_name = str(i) + '.png'
        imsave(path_to_save + img_name, img)

        if (i + 1) % 500 == 0:
            print(i)

        i += 1

print(i, 'done')
