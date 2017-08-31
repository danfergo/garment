import os
from PIL import Image
import numpy as np


def open_images(folder, expand_dims=True, n=-1):
    images = []
    files = os.listdir(folder)
    files.sort()
    i = 1
    for filename in files:
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            arr_img = np.array(img)
            if expand_dims:
                if len(arr_img.shape) > 2:
                    images.append(arr_img)
                else:
                    s = arr_img.shape
                    arr2_img = np.zeros((s[0], s[1], 3))
                    arr2_img[:, :, 0] = arr_img
                    arr2_img[:, :, 1] = arr_img
                    arr2_img[:, :, 2] = arr_img

                    images.append(arr2_img)
            else:
                images.append(arr_img)
        i += 1
        if 0 < n < i:
            return images
    return images
