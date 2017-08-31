# /home/danfergo/SIG/Datasets/CloPeMa/CTU/ColorAndDepth/FlatAndWrinkled
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

import random
from PIL import Image

import os
import shutil
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import yaml

WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX
FW = 1
LH = 20
FS = 0.5

DATASET_PATH = '../../../Datasets/DeepFashion/'
EVAL_FILE = '../../../Datasets/DeepFashion/eval/list_eval_partition.txt'
ANNO_CAT_FILE = '../../../Datasets/DeepFashion/anno/list_category_img.txt'
ANNO_CAT_NAMES = '../../../Datasets/DeepFashion/anno/list_category_cloth.txt'

OUT_PATH = '../data/deepfashion/'

OUT = {
    'train': OUT_PATH + 'train/',
    'val': OUT_PATH + 'validation/',
    'test': OUT_PATH + 'test/'
}

VALIDATION_PERCENT = 0.1


valid_counter = 0
counter_per_cat = {}


dataset = {}
categories = {}


def main():
    clean_folder()
    load_anno_cat_names()
    load_anno_cat()
    load_eval()
    read_raw_dataset()
    show_stats()


def clean_folder():
    random.seed()

    shutil.rmtree(OUT_PATH)
    for env in OUT:
        os.makedirs(OUT[env])


def load_anno_cat_names():
    global categories

    f = open(ANNO_CAT_NAMES, 'r')
    data = f.readlines()

    i = 0
    id = 1
    for line in data:
        if i < 2:
            i += 1
            continue
        line_arr = line.split()
        categories[str(id)] = {
            'name': line_arr[0],
            'type': line_arr[1]
        }
        id += 1


def load_anno_cat():
    global dataset

    f = open(ANNO_CAT_FILE, 'r')
    data = f.readlines()

    i = 0
    for line in data:
        if i < 2:
            i += 1
            continue
        line_arr = line.split()
        dataset[line_arr[0]] = {
            'category_id': line_arr[1]
        }


def load_eval():
    global dataset

    f = open(EVAL_FILE, 'r')
    data = f.readlines()

    i = 0
    for line in data:
        if i < 2:
            i += 1
            continue
        line_arr = line.split()
        dataset[line_arr[0]]['eval'] = line_arr[1]


def read_raw_dataset():
    global valid_counter, dataset, categories

    for file_name in dataset:
            try:
                # stream = open(path + file_name)
                # meta_data = yaml.load(stream)

                # img_file_name = meta_data['path_c']
                # img_category = meta_data['type']
                img_category_id = dataset[file_name]['category_id']
                img_category = categories[img_category_id]['name']
                out_path = OUT[dataset[file_name]['eval']]
                img = load_img(DATASET_PATH + file_name, target_size=(150, 150))
                # img = img.resize((320, 256), Image.ANTIALIAS)

                x = img_to_array(img)

                if img_category in counter_per_cat:
                    counter_per_cat[img_category] += 1
                else:
                    counter_per_cat[img_category] = 1
                valid_counter += 1

                if not os.path.exists(out_path + img_category):
                     os.makedirs(out_path + img_category)

                img.save(out_path +
                         '/' + img_category +
                         '/' + str(counter_per_cat[img_category]).zfill(4) + '.jpeg')

                y = x.astype(np.uint8)
                y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
                cv2.imshow("image", y)
                cv2.setWindowTitle("image",
                                   'Ttl: ' + str(valid_counter).zfill(4).ljust(10) +
                                   'Cat: ' + img_category.ljust(15) +
                                   '/cat: ' + str(counter_per_cat[img_category]).zfill(3).ljust(6))
                cv2.waitKey(1)

            except Exception as e:
                # print str(e)
                continue


def show_stats():
    stats_frame = np.zeros((256, 320), dtype=np.uint8)

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



# x = x.reshape((1,) + x.shape)
# i = 0
# for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
#         i += 1
# cv2.imshow("image", batch[0].astype(np.uint8))
# cv2.waitKey(2)

# if i > 2:
# break  # otherwise the generator would loop indefinitely
