import Image
import PIL

import random

import os
import shutil
# import cv2
import scipy.io
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from scipy.misc import imresize

DATASET_PATH = '../../../Datasets/clothing-co-parsing/'
PHOTOS_PATH = DATASET_PATH + 'photos/'
ANNO_IMG_PATH = DATASET_PATH + 'annotations/image-level/'
ANNO_PIXEL_PATH = DATASET_PATH + 'annotations/pixel-level/'
LABEL_LIST_FILE = DATASET_PATH + 'label_list.mat'

OUT_PATH = '../data/ccp/'
OUT_SUB = {
    'train': 'train/',
    'validation': 'validation/',
    'test': 'test/'
}

OUT_PERCENT = {
    'train': 0.8,
    'validation': 0.1,
    'test': 0.1
}


COLOR_MAP = [
    (73, 38, 223),  # null
    (37, 0, 147),  # accessories
    (33, 1, 128),  # bag
    (256, 99, 47),  # belt
    (0, 0, 1),  # blazer
    (257, 2, 199),  # blouse
    (0, 0, 2),  # bodysuit
    (60, 3, 222),  # boots
    (0, 0, 3),  # bra
    (0, 0, 4),  # bracelet
    (70, 1, 239),  # cape
    (0, 0, 5),  # cardigan
    (0, 0, 6),  # clogs
    (42, 1, 159),  # coat
    (73, 82, 239),  # dress
    (0, 0, 7),  # earrings
    (0, 0, 8),  # flats
    (0, 0, 9),  # glasses
    (0, 0, 10),  # gloves
    (82, 195, 235),  # hair
    (81, 216, 235),  # hat
    (0, 0, 11),  # heels
    (0, 0, 12),  # hoodie
    (0, 0, 13),  # intimate
    (87, 255, 206),  # jacket
    (91, 255, 192),  # jeans
    (0, 0, 14),  # jumper
    (0, 0, 15),  # leggings
    (125, 253, 140),  # loafers
    (0, 0, 16),  # necklace
    (0, 0, 17),  # panties
    (154, 255, 82),  # pants
    (0, 0, 18),  # pumps
    (193, 248, 67),  # purse
    (0, 0, 19),  # ring
    (0, 0, 20),  # romper
    (241, 254, 14),  # sandals
    (0, 0, 21),  # scarf
    (249, 255, 1),  # shirt
    (255, 238, 0),  # shoes
    (0, 0, 22),  # shorts
    (255, 197, 1),  # skin
    (254, 197, 0),  # skirt
    (0, 0, 23),  # sneakers
    (0, 0, 24),  # socks
    (255, 123, 0),  # stockings
    (0, 0, 25),  # suit
    (21, 90, 0),  # sunglasses
    (255, 71, 0),  # sweater
    (0, 0, 26),  # sweatshirt
    (0, 0, 27),  # swimwear
    (248, 20, 7),  # t-shirt
    (244, 24, 2),  # tie
    (0, 0, 28),  # tights
    (0, 0, 29),  # top
    (0, 0, 30),  # vest
    (177, 14, 5),  # wallet
    (0, 0, 31),  # watch
    (108, 0, 1)  # wedges
]

O_SIZE = (150,150)

# img_anno = scipy.io.loadmat(ANNO_IMG_LVL + str(1005).zfill(4) + '.mat',  squeeze_me=True)
# pixel_anno = scipy.io.loadmat(ANNO_PIXEL_PATH + str(1).zfill(4) + '.mat',  squeeze_me=True).get('groundtruth')


LABEL_LIST = scipy.io.loadmat(LABEL_LIST_FILE, squeeze_me=True).get('label_list')



def build_paths(out, xy, sub):
    paths = {}
    for k in sub:
        paths[k] = out + xy + '/' + sub[k]
    return paths


def init_folders(out, folders):
    shutil.rmtree(out)
    for fg in folders:
        for e in fg:
            os.makedirs(fg[e])


def read_raw_dataset(x_paths, y_paths, out_percent):
    for x in range(1, 1001):
        # img_anno_file = ANNO_IMG_LVL + str(1005).zfill(4)
        # img_anno = scipy.io.loadmat(img_anno_file + '.mat',  squeeze_me=True)
        # print img_anno

        pixel_anno_file = ANNO_PIXEL_PATH + str(x).zfill(4) + '.mat'
        pil_bw_pixel_anno_f = scipy.io.loadmat(pixel_anno_file, squeeze_me=True)
        pil_bw_pixel_anno = pil_bw_pixel_anno_f.get('groundtruth')
        pil_bw_pixel_anno = imresize(pil_bw_pixel_anno, O_SIZE, 'nearest')
        bw_pixel_anno = img_to_array(pil_bw_pixel_anno)

        # map annotation to color
        height, width, depth = bw_pixel_anno.shape
        color_anno = np.zeros((height, width, 3), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                cat = int(bw_pixel_anno[i, j][0])
                if cat > 58:
                    print cat
                color_anno[i, j] = COLOR_MAP[cat]

        # selecting dataset
        dataset = ''
        accumulator = 0
        rnd = random.random()
        for e in out_percent:
            accumulator += out_percent[e]
            if rnd < accumulator:
                dataset = e
                break

        # load image 275, 415
        pil_image = load_img(PHOTOS_PATH + str(x).zfill(4) + '.jpg')
        pil_image = pil_image.resize(O_SIZE, Image.ANTIALIAS)
        image = img_to_array(pil_image)

        pil_anno = Image.fromarray(pil_bw_pixel_anno.astype(np.uint8), mode='L')
        pil_image.save(x_paths[dataset] + str(x).zfill(4) + '.jpeg')
        pil_anno.save(y_paths[dataset] + str(x).zfill(4) + '.bmp', mode='L')

        # show
        # cv2.namedWindow("image")
        # cv2.moveWindow("image", 0, 0)
        # cv2.setWindowTitle("image", str(x).zfill(4))
        # image = image.astype(np.uint8)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("image", image)
        #
        # cv2.namedWindow("anno")
        # cv2.moveWindow("anno", width, 0)
        # color_anno = cv2.cvtColor(color_anno, cv2.COLOR_RGB2BGR)
        # cv2.imshow("anno", color_anno)
        #
        # cv2.waitKey(1)


def main():
    x_paths = build_paths(OUT_PATH, 'x', OUT_SUB)
    y_paths = build_paths(OUT_PATH, 'y', OUT_SUB)
    init_folders(OUT_PATH, [x_paths, y_paths])
    read_raw_dataset(x_paths, y_paths, OUT_PERCENT)


if __name__ == '__main__':
    main()

