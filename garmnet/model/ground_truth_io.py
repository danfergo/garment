import random

import numpy as np
import hashlib
import os.path
import pickle
from lib.util.math import coord_to_box, iou


class LossLoader:
    G_C = 0  # garment class
    G_BB = 2  # garment bounding-box
    G_X = 0  # garment bounding-box x
    G_Y = 1  # garment bounding-box y
    G_W = 1  # garment bounding-box with
    G_H = 1  # garment bounding-box height

    L = 1  # landmark
    L_C = 1  # landmark class
    L_X = 1  # landmark bounding-box x
    L_Y = 1  # landmark bounding-box y

    def __first_not_none(self, tuple):
        for t in tuple:
            if t is not None:
                return t
        print('last is none')

    def __init__(self, n_garment_cats, n_landmark_cats, bboxes_configs, low_t=0.3, high_t=0.7):
        self.low_t = low_t
        self.high_t = high_t
        self.bboxes_configs = bboxes_configs
        self.n_landmark_cats = n_landmark_cats
        self.n_garment_cats = n_garment_cats

    def as_output(self, data):
        landmarks_cls, landmarks_reg, cls, reg = data
        n_examples = self.__first_not_none(data).shape[0]
        bb_size, stride, am_size = self.bboxes_configs

        output_data = []

        for x in range(n_examples):

            landmarks = []
            if landmarks_cls is not None:
                for i in range(landmarks_reg.shape[1]):
                    for j in range(landmarks_reg.shape[2]):
                        c = np.argmax(landmarks_cls[x, i, j])
                        confidence = landmarks_cls[x, i, j, c]
                        # if confidence > 0.5:
                        landmarks.append([
                            stride * j + landmarks_reg[x, i, j, 0] + bb_size / 2,
                            stride * i + landmarks_reg[x, i, j, 1] + bb_size / 2,
                            c,
                            confidence
                        ])

            c = np.argmax(cls[x]) if cls is not None else None
            bb = reg[x] * 224 if reg is not None else None
            output_data.append([
                c,
                bb,
                landmarks,
                # []
                landmarks
            ])

        return output_data

    def as_input(self, data):
        debug = False
        bb_size, stride, am_size = self.bboxes_configs

        # print(data.shape)
        data_x, data_y = data
        length = data_x.shape[0]
        img_size = data_x[0].shape

        # for bb in range(0, 225):
        #     for s in range(1, 225):
        #         if (img_size[0] - bb) % s == 0 and ((img_size[0] - bb) / s) + 1 == am_size[0]:
        #             print('bounding-box size: ' + str(bb) + ', stride: ' + str(s) + ' ratio: ' + str(bb/s))

        assert ((img_size[0] - bb_size) % stride == 0)
        assert (((img_size[0] - bb_size) / stride) + 1 == am_size[0])

        # pi: fg, bg
        landmark_cls = np.zeros((length,) + am_size + (self.n_landmark_cats + 1,))

        # ti: tx, ty, tw, th, cls[0]
        landmark_reg = np.zeros((length,) + am_size + (3,))

        cls = np.zeros((length,) + (self.n_garment_cats,))

        reg = np.zeros((length,) + (4,))

        # to produce stats
        negatives = float(0)
        positives = float(0)
        neutrals = float(0)

        # for all images length
        for x in range(length):

            # for all ground truths
            gts = np.array(data_y[x][2])
            gts_ok_max = {}
            gts_ok_max_owner = {}

            for i in range(0, am_size[0]):
                for j in range(0, am_size[1]):
                    box = coord_to_box((j * stride, i * stride), bb_size)

                    s_gts = sorted(enumerate(gts), key=lambda g: iou(coord_to_box(g[1] - bb_size / 2, bb_size), box),
                                   reverse=True)

                    gt = coord_to_box(s_gts[0][1] - bb_size / 2, bb_size)
                    gt_iou = iou(gt, box)

                    c = s_gts[0][1][2]

                    # filling rpn and roi cls
                    if gt_iou >= self.high_t:
                        landmark_cls[x, i, j, c] = 1
                        # roi_cls[x, i, j, c] = 1
                        positives += 1
                    elif gt_iou < self.low_t:
                        landmark_cls[x, i, j, 0] = 1  # [1, 0] background
                        # roi_cls[x, i, j, 0] = 1  # background
                        negatives += 1
                    else:
                        neutrals += 1

                    k = str(s_gts[0][0])
                    if k not in gts_ok_max or gt_iou > gts_ok_max[k]:
                        gts_ok_max[k] = gt_iou
                        gts_ok_max_owner[k] = [i, j, c]

                    landmark_reg[x, i, j, 0] = float(gt[0] - box[0])  # (2 * delta)   tx
                    landmark_reg[x, i, j, 1] = float(gt[1] - box[1])  # (2 * delta)   ty

                    is_landmark = 0 if np.argmax(landmark_cls[x, i, j]) == 0 else 1
                    landmark_reg[x, i, j, 2] = is_landmark
                    landmark_cls[x, i, j, self.n_landmark_cats] = is_landmark

            # setting up the closest
            for k in gts_ok_max:
                if gts_ok_max[k] < self.high_t:
                    i = gts_ok_max_owner[k][0]
                    j = gts_ok_max_owner[k][1]
                    c = gts_ok_max_owner[k][2]

                    # landmark_cls[x, i, j] = [0, 1]  # foreground
                    landmark_cls[x, i, j, c] = 1
                    landmark_reg[x, i, j, 2] = 1
                    landmark_cls[x, i, j, self.n_landmark_cats] = 1

            # turn on 10 negatives
            n = 0
            while n < 10:
                i = random.randint(0, am_size[0] - 1)
                j = random.randint(0, am_size[1] - 1)

                if landmark_cls[x, i, j, self.n_landmark_cats] == 0 and landmark_cls[x, i, j, 0] == 1:
                    landmark_cls[x, i, j, self.n_landmark_cats] = 1
                    n += 1

            cls[x, data_y[x][0]] = 1
            reg[x] = np.array(data_y[x][1]) / img_size[0]

        # roi_cls_s = roi_cls.shape
        # roi_reg_s = roi_reg.shape
        # roi_cls = np.reshape(roi_cls, (roi_cls_s[0], roi_cls_s[1] * roi_cls_s[2], roi_cls_s[3]))
        # roi_reg = np.reshape(roi_reg, (roi_reg_s[0], roi_reg_s[1] * roi_reg_s[2], roi_reg_s[3]))

        # with open('cache/.' + cache_file_name, 'wb') as f:
        #     pickle.dump((train_x, rpn_cls, rpn_reg, roi_cls, roi_reg, cls, reg), f)

        # print stats
        total = positives + negatives + neutrals
        if debug:
            print('-------------------------------------------------')
            print('Size: ' + str(length))
            print('ANCHORS: ')
            print('Landmark bounding-box size: ' + str(bb_size))
            print('Positives mean ratio: ' + str(positives / total))
            print('Negatives mean ratio: ' + str(negatives / total))
            print('Neutrals mean ratio: ' + str(neutrals / total))
            print('-------------------------------------------------')

        return landmark_cls, landmark_reg, cls, reg

    def re_output(self, data):
        return self.as_output(self.as_input(data))


def main():
    from lib.datasets.clopema import ClopemaLoader
    from model.garmnet import GarmNet
    from lib.util.display import Display
    import keras.backend as k

    with k.get_session():
        train_name = 'with_bridge'
        use_bridge = True

        dataset = ClopemaLoader()
        m = GarmNet(dataset.image_shape(), dataset.n_garment_cats(), dataset.n_landmark_cats(), use_bridge)
        display = Display(dataset, m.bb_size)

        data = dataset.fetch_data('val')
        data_x, data_y = data

        display.show_multiple_results(data_x, None,
                                      annotations=data_y,
                                      rpn_ground_truths=m.get_rpn_ground_truths(data))


if __name__ == '__main__':
    main()
