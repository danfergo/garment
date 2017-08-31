import random
import os
import shutil
# import cv2
import numpy as np
from PIL import Image
import yaml
import pickle as pkl

SAVE_DATASET = True

WHITE = (255, 255, 255)
# FONT = cv2.FONT_HERSHEY_DUPLEX
FW = 1
LH = 25
FS = 0.6

PATH_FLAT_N_WRINKLED = '/home/danfergo/SIG/Datasets/CloPeMa/CTU/ColorAndDepth/FlatAndWrinkled/'
PATH_FOLDED = '/home/danfergo/SIG/Datasets/CloPeMa/CTU/ColorAndDepth/Folded/'

OUT_PATH = '/home/danfergo/SIG/Code/Experiments/data/clopema/'
OUT_TRAIN_PATH = OUT_PATH + 'train/'
OUT_VALIDATION_PATH = OUT_PATH + 'validation/'

VALIDATION_PERCENT = 0.1
# RS_FX = 4
# RS_FY = 5

IMG_SIZE = (224, 224)
RS_FX = float(1024) / IMG_SIZE[0]
RS_FY = float(1280) / IMG_SIZE[1]

valid_counter = 0
counter_per_cat = {}

LANDMARK_DELTA = 5

# IMG_SIZE = (int(1280 / RS_FY), int(1024 / RS_FX))

landmark_colors = []
landmark_names = []

landmark_names.append('null')
landmark_colors.append((0, 0, 0))

# pants
landmark_names.append('left-leg-outer')
landmark_colors.append((0, 0, 200))
landmark_names.append('left-leg-inner')
landmark_colors.append((0, 0, 255))
landmark_names.append('crotch')
landmark_colors.append((0, 255, 150))
landmark_names.append('right-leg-inner')
landmark_colors.append((0, 255, 0))
landmark_names.append('rigth-leg-outer')
landmark_colors.append((0, 200, 0))
landmark_names.append('top-right')
landmark_colors.append((0, 200, 200))
landmark_names.append('top-left')
landmark_colors.append((0, 255, 255))

# tshirt
# 'bottom-left':
# 'bottom-right':
# 'right-armpit':
landmark_names.append('right-sleave-inner')
landmark_colors.append((255, 255, 0))
landmark_names.append('right-sleave-outer')
landmark_colors.append((150, 150, 0))
# 'right-shoulder':
# 'neckline-right':
# 'neckline-left':
# 'left-shoulder':
landmark_names.append('left-sleave-outer')
landmark_colors.append((255, 255, 255))
landmark_names.append('left-sleave-inner')
landmark_colors.append((200, 200, 200))
# 'left-armpit':

# tshirt-long
# 'bottom-left':
# 'bottom-right':
# 'right-armpit':
# 'right-sleave-inner':
# 'right-sleave-outer':
# 'right-shoulder':
# 'neckline-right':
# 'neckline-left':
# 'left-shoulder':
# 'left-sleave-outer':
# 'left-sleave-inner':
# 'left-armpit':


# towel
# 'bottom-left')
# landmark_colors.append((255, 255, 0))
# 'bottom-right')
# landmark_colors.append((200, 200, 0))
# 'top-right')
# landmark_colors.append((150, 150, 0))
# 'top-left')
# landmark_colors.append((100, 100, 0))

# skirt
# 'bottom-left':
# 'bottom-right':
# 'top-right':
# 'top-left':

# hoody
# 'bottom-left':
# 'bottom-right':
# 'right-armpit':
# 'right-sleave-inner':
# 'right-sleave-outer':
# 'right-shoulder':
landmark_names.append('hood-right')
landmark_colors.append((255, 0, 255))
landmark_names.append('hood-top')
landmark_colors.append((200, 0, 200))
landmark_names.append('hood-left')
landmark_colors.append((150, 0, 150))
# 'left-shoulder':
# 'left-sleave-outer':
# 'left-sleave-inner':
# 'left-armpit':

# polo
# 'bottom-left':
# 'bottom-right':
# 'right-armpit':
# 'right-sleave-inner':
# 'right-sleave-outer':
# 'right-shoulder':
# 'neckline-right':
# 'collar-rigth')
# landmark_colors.append((200, 0, 200))
# 'collar-left')
# landmark_colors.append((150, 0, 150))
# 'neckline-left':
# 'left-shoulder':
# 'left-sleave-outer':
# 'left-sleave-inner':
# 'left-armpit':

# polo-long
# 'bottom-left':
# 'bottom-right':
# 'right-armpit':
# 'right-sleave-inner':
# 'right-sleave-outer':
# 'right-shoulder':
# 'neckline-right':
# 'collar-rigth':
# 'collar-left':
# 'neckline-left':
# 'left-shoulder':
# 'left-sleave-outer':
# 'left-sleave-inner':
# 'left-armpit':

# bluse
# 'bottom-left':
# 'bottom-middle':
# 'bottom-right':
# 'right-armpit':
# 'right-shoulder':
# 'neckline-right':
# 'collar-right':
# 'collar-left':
# 'neckline-left':
# 'left-shoulder':
# 'left-armpit':

# bluse
landmark_names.append('bottom-left')
landmark_colors.append((0, 0, 150))
landmark_names.append('bottom-middle')
landmark_colors.append((0, 0, 200))
landmark_names.append('bottom-right')
landmark_colors.append((0, 0, 255))
landmark_names.append('right-armpit')
landmark_colors.append((0, 255, 0))
landmark_names.append('right-shoulder')
landmark_colors.append((0, 200, 0))
landmark_names.append('neckline-right')
landmark_colors.append((0, 250, 0))
landmark_names.append('collar-right')
landmark_colors.append((0, 255, 250))
landmark_names.append('collar-left')
landmark_colors.append((0, 255, 255))
landmark_names.append('neckline-left')
landmark_colors.append((0, 150, 0))
landmark_names.append('left-shoulder')
landmark_colors.append((0, 200, 0))
landmark_names.append('left-armpit')
landmark_colors.append((0, 0, 255))

landmark_names.append('fold_1')
landmark_colors.append((0, 200, 255))
landmark_names.append('fold_2')
landmark_colors.append((0, 150, 200))

assert(len(landmark_names) == len(landmark_colors))
print('n landmarks' + str(len(landmark_names)))

COLOR_MAP = {
    # pants
    'left-leg-outer': (0, 0, 200),
    'left-leg-inner': (0, 0, 255),
    'crotch': (0, 255, 150),
    'right-leg-inner': (0, 255, 0),
    'rigth-leg-outer': (0, 200, 0),
    'top-right': (0, 200, 200),
    'top-left': (0, 255, 255),

    # tshirt
    # 'bottom-left':
    # 'bottom-right':
    # 'right-armpit':
    'right-sleave-inner': (255, 255, 0),
    'right-sleave-outer': (150, 150, 0),
    # 'right-shoulder':
    # 'neckline-right':
    # 'neckline-left':
    # 'left-shoulder':
    'left-sleave-outer': (255, 255, 255),
    'left-sleave-inner': (200, 200, 200),
    # 'left-armpit':

    # tshirt-long
    # 'bottom-left':
    # 'bottom-right':
    # 'right-armpit':
    # 'right-sleave-inner':
    # 'right-sleave-outer':
    # 'right-shoulder':
    # 'neckline-right':
    # 'neckline-left':
    # 'left-shoulder':
    # 'left-sleave-outer':
    # 'left-sleave-inner':
    # 'left-armpit':


    # towel
    # 'bottom-left': (255, 255, 0),
    # 'bottom-right': (200, 200, 0),
    # 'top-right': (150, 150, 0),
    # 'top-left': (100, 100, 0),

    # skirt
    # 'bottom-left':
    # 'bottom-right':
    # 'top-right':
    # 'top-left':

    # hoody
    # 'bottom-left':
    # 'bottom-right':
    # 'right-armpit':
    # 'right-sleave-inner':
    # 'right-sleave-outer':
    # 'right-shoulder':
    'hood-right': (255, 0, 255),
    'hood-top': (200, 0, 200),
    'hood-left': (150, 0, 150),
    # 'left-shoulder':
    # 'left-sleave-outer':
    # 'left-sleave-inner':
    # 'left-armpit':

    # polo
    # 'bottom-left':
    # 'bottom-right':
    # 'right-armpit':
    # 'right-sleave-inner':
    # 'right-sleave-outer':
    # 'right-shoulder':
    # 'neckline-right':
    # 'collar-rigth': (200, 0, 200),
    # 'collar-left': (150, 0, 150),
    # 'neckline-left':
    # 'left-shoulder':
    # 'left-sleave-outer':
    # 'left-sleave-inner':
    # 'left-armpit':

    # polo-long
    # 'bottom-left':
    # 'bottom-right':
    # 'right-armpit':
    # 'right-sleave-inner':
    # 'right-sleave-outer':
    # 'right-shoulder':
    # 'neckline-right':
    # 'collar-rigth':
    # 'collar-left':
    # 'neckline-left':
    # 'left-shoulder':
    # 'left-sleave-outer':
    # 'left-sleave-inner':
    # 'left-armpit':

    # bluse
    # 'bottom-left':
    # 'bottom-middle':
    # 'bottom-right':
    # 'right-armpit':
    # 'right-shoulder':
    # 'neckline-right':
    # 'collar-right':
    # 'collar-left':
    # 'neckline-left':
    # 'left-shoulder':
    # 'left-armpit':

    # bluse
    'bottom-left': (0, 0, 150),
    'bottom-middle': (0, 0, 200),
    'bottom-right': (0, 0, 255),
    'right-armpit': (0, 255, 0),
    'right-shoulder': (0, 200, 0),
    'neckline-right': (0, 250, 0),
    'collar-right': (0, 255, 250),
    'collar-left': (0, 255, 255),
    'neckline-left': (0, 150, 0),
    'left-shoulder': (0, 200, 0),
    'left-armpit': (0, 0, 255),

    'fold_1': (0, 200, 255),
    'fold_2': (0, 150, 200),

}

data_x_tmp = []
data_y_tmp = []
data_x = {'VAL': [], 'TRAIN': []}
data_y = {'VAL': [], 'TRAIN': []}

categories = ['pants', 'polo', 'hoody', 'tshirt', 'tshirt-long', 'polo-long', 'towel', 'skirt', 'bluse']


def main():
    global data_x_tmp, data_y_tmp

    read_raw_dataset()
    show_stats()

    data_x_tmp = np.array(data_x_tmp)
    data_y_tmp = np.array(data_y_tmp, dtype=object)

    indices = np.random.permutation(data_x_tmp.shape[0])
    training_idx, test_idx = indices[300:], indices[:300]

    train_x, val_x = data_x_tmp[training_idx], data_x_tmp[test_idx]
    train_y, val_y = data_y_tmp[training_idx], data_y_tmp[test_idx]

    if SAVE_DATASET:
        clean_folder()
        np.save(OUT_TRAIN_PATH + '/data_x', train_x)
        np.save(OUT_TRAIN_PATH + '/data_y', train_y)

        np.save(OUT_VALIDATION_PATH + '/data_x', val_x)
        np.save(OUT_VALIDATION_PATH + '/data_y', val_y)

        # np.save(OUT_TRAIN_PATH + '/data_x', np.array(data_x['TRAIN']))
        # np.save(OUT_TRAIN_PATH + '/data_y', np.array(data_y['TRAIN'], dtype=object))
        #
        # np.save(OUT_VALIDATION_PATH + '/data_x', np.stack(data_x['VAL']))
        # np.save(OUT_VALIDATION_PATH + '/data_y', np.array(data_y['VAL'], dtype=object))


def clean_folder():
    random.seed()

    shutil.rmtree(OUT_PATH)
    os.makedirs(OUT_TRAIN_PATH)
    os.makedirs(OUT_VALIDATION_PATH)


def read_raw_dataset():
    global valid_counter, data_x, data_y, categories

    for g in range(1, 3332):
        # 3332

        path = PATH_FLAT_N_WRINKLED if g < 2051 else PATH_FOLDED

        try:
            file_name = 'cloA' + str(g).zfill(5) + '.yaml'
            meta_file = open(path + file_name)
            meta_data = yaml.load(meta_file)

            img_file_name = meta_data['path_c']
            img_category = meta_data['type']

            img_poly = np.array(meta_data['poly_c'])
            img_node_names = np.array(meta_data['node_names'])

            img_pil = Image.open(path + img_file_name)
            img_pil = img_pil.resize(IMG_SIZE, Image.ANTIALIAS)

            img = np.asarray(img_pil)

            # if img_category not in categories:
            #     categories.append(img_category)

            if img_category in counter_per_cat:
                counter_per_cat[img_category] += 1
            else:
                counter_per_cat[img_category] = 1
            valid_counter += 1

            # sub_set = 'VAL' if random.random() < VALIDATION_PERCENT else 'TRAIN'

            # if SAVE_DATASET:
            #     path = OUT_VALIDATION_PATH if sub_set == 'VAL' else OUT_TRAIN_PATH
            # if not os.path.exists(path + img_category):
            #     os.makedirs(path + img_category)

            landmarks = []
            for i in range(img_poly.shape[0]):
                x = (int(img_poly[i, 0] / RS_FY))
                y = (int(img_poly[i, 1] / RS_FX))

                # y1 = (y - LANDMARK_DELTA)
                # x1 = (x - LANDMARK_DELTA)
                # h = (y + LANDMARK_DELTA) - y1
                # w = (x + LANDMARK_DELTA) - x1

                # dataset typos
                if img_node_names[i, 0] == 'collar-rigth':
                    img_node_names[i, 0] = 'collar-right'

                landmark_id = landmark_names.index(img_node_names[i, 0])
                landmarks.append([x, y, landmark_id])

            x1 = int(np.amin(img_poly[:, 0]) / RS_FY)
            y1 = int(np.amin(img_poly[:, 1]) / RS_FX)
            x2 = int(np.amax(img_poly[:, 0]) / RS_FY)
            y2 = int(np.amax(img_poly[:, 1]) / RS_FX)

            data_x_tmp.append(img)
            data_y_tmp.append([categories.index(img_category), [x1, y1, x2 - x1, y2 - y1], landmarks])

            # plotting landmarks
            int_img = img.astype(np.uint8)

            for i in range(img_poly.shape[0]):
                y = (int(img_poly[i, 0] / RS_FY))
                x = (int(img_poly[i, 1] / RS_FX))

                x1 = (y - LANDMARK_DELTA)
                y1 = (x - LANDMARK_DELTA)
                x2 = (y + LANDMARK_DELTA)
                y2 = (x + LANDMARK_DELTA)

                # dataset typos
                if img_node_names[i, 0] == 'collar-rigth':
                    img_node_names[i, 0] = 'collar-right'

                landmark_id = landmark_names.index(img_node_names[i, 0])
                # cv2.rectangle(int_img, (x1, y1), (x2, y2), landmark_colors[landmark_id], 1)
                # cv2.circle(int_img, (int(m[0': / RS_F), int(m[1': / RS_F)), 5, (0, 0, 255), 1)

            # calculate
            x1 = int(np.amin(img_poly[:, 0]) / RS_FY)
            y1 = int(np.amin(img_poly[:, 1]) / RS_FX)
            x2 = int(np.amax(img_poly[:, 0]) / RS_FY)
            y2 = int(np.amax(img_poly[:, 1]) / RS_FX)

            # print('Ttl: ' + str(valid_counter).zfill(4).ljust(10) +
            #       'Cat: ' + img_category.ljust(15) +
            #       '/cat: ' + str(counter_per_cat[img_category]).zfill(3).ljust(6))

            # cv2.rectangle(int_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            #
            # # displaying image with header
            # cv2.imshow("image", int_img)
            # cv2.setWindowTitle("image",
            #                    'Ttl: ' + str(valid_counter).zfill(4).ljust(10) +
            #                    'Cat: ' + img_category.ljust(15) +
            #                    '/cat: ' + str(counter_per_cat[img_category]).zfill(3).ljust(6))
            # cv2.waitKey(1)

        except Exception as e:
            # print(str(e))
            continue


def show_stats():
    pass
    # stats_frame = np.zeros(IMG_SIZE, dtype=np.uint8)
    #
    # i = 1
    # for key in counter_per_cat:
    #     cv2.putText(stats_frame, key, (10, i * LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
    #     cv2.putText(stats_frame, str(counter_per_cat[key]), (100, i * LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
    #     i += 1
    #     cv2.putText(stats_frame, 'TOTAL', (10, (i + 1) * LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
    #     cv2.putText(stats_frame, str(valid_counter), (100, (i + 1) * LH), FONT, FS, WHITE, FW, cv2.LINE_AA)
    #
    # cv2.imshow("image", stats_frame)
    # cv2.setWindowTitle("image", "Stats")
    # cv2.waitKey(10000)


if __name__ == '__main__':
    main()
