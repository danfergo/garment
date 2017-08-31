import random
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import yaml

SAVE_DATASET = True


WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX
FW = 1
LH = 25
FS = 0.6

PATH_FLAT_N_WRINKLED = '../../../Datasets/CloPeMa/CTU/ColorAndDepth/FlatAndWrinkled/'
PATH_FOLDED = '../../../Datasets/CloPeMa/CTU/ColorAndDepth/Folded/'

OUT_PATH = '../data/clopema/'
OUT_TRAIN_PATH = OUT_PATH + 'train/'
OUT_VALIDATION_PATH = OUT_PATH + 'validation/'

VALIDATION_PERCENT = 0.1
RS_F = 3


valid_counter = 0
counter_per_cat = {}

threshold = 1160


def main():
    if SAVE_DATASET:
        clean_folder()

    read_raw_dataset()
    show_stats()


def clean_folder():
    random.seed()

    shutil.rmtree(OUT_PATH)
    os.makedirs(OUT_TRAIN_PATH)
    os.makedirs(OUT_VALIDATION_PATH)


def read_raw_dataset():
    global valid_counter

    for x in range(1, 3332):

            path = PATH_FLAT_N_WRINKLED if x < 2051 else PATH_FOLDED

            try:
                file_name = 'cloA' + str(x).zfill(5) + '.yaml'
                meta_file = open(path + file_name)
                meta_data = yaml.load(meta_file)

                img_file_name = meta_data['path_c']
                img_category = meta_data['type']
                img_poly = meta_data['poly_c']

                img_pil = Image.open(path + img_file_name)
                img_pil = img_pil.resize((1280/RS_F, 1024/RS_F), Image.ANTIALIAS)

                img = np.asarray(img_pil)

                if img_category in counter_per_cat:
                    counter_per_cat[img_category] += 1
                else:
                    counter_per_cat[img_category] = 1
                valid_counter += 1

                if SAVE_DATASET:
                    path = OUT_VALIDATION_PATH if random.random() < VALIDATION_PERCENT else OUT_TRAIN_PATH
                    if not os.path.exists(path + img_category):
                        os.makedirs(path + img_category)

                    img_pil.save(path +
                             '/' + img_category +
                             '/' + str(counter_per_cat[img_category]).zfill(4) + '.jpeg')

                int_img = img.astype(np.uint8)
                for m in img_poly:
                    cv2.circle(int_img, (int(m[0]/RS_F), int(m[1]/RS_F)), 5, (0, 0, 255), 1)

                cv2.imshow("image", int_img)
                cv2.setWindowTitle("image",
                                   'Ttl: ' + str(valid_counter).zfill(4).ljust(10) +
                                   'Cat: ' + img_category.ljust(15) +
                                   '/cat: ' + str(counter_per_cat[img_category]).zfill(3).ljust(6))
                cv2.waitKey(1)

            except Exception as e:
                print(str(e))
                continue


def show_stats():
    stats_frame = np.zeros((1024/RS_F, 1280/RS_F), dtype=np.uint8)

    i = 1
    for key in counter_per_cat:
        cv2.putText(stats_frame, key, (10, i*LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
        cv2.putText(stats_frame, str(counter_per_cat[key]), (100, i*LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
        i += 1
    cv2.putText(stats_frame, 'TOTAL', (10, (i+1) * LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
    cv2.putText(stats_frame, str(valid_counter), (100, (i+1) * LH), FONT, FS, WHITE, FW, cv2.LINE_AA)

    cv2.imshow("image", stats_frame)
    cv2.setWindowTitle("image", "Stats")
    cv2.waitKey(10000)


if __name__ == '__main__':
    main()

