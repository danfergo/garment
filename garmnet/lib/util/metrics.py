import numpy as np

from sklearn.metrics import accuracy_score
from lib.datasets.clopema import ClopemaLoader
from lib.util.math import iou
from model.ground_truth_io import LossLoader

D = 26
epsilon = 1e-4
iou_threshold = 0.5


def tensor_iou(lib, a, b):
    i_dx = lib.maximum(lib.minimum(a[1] + D, b[1] + D) - lib.maximum(a[1], b[1]), 0)
    i_dy = lib.maximum(lib.minimum(a[0] + D, b[0] + D) - lib.maximum(a[0], b[0]), 0)
    i = i_dx * i_dy

    u_dx = lib.maximum(lib.maximum(a[1] + D, b[1] + D) - lib.minimum(a[1], b[1]), 0)
    u_dy = lib.maximum(lib.maximum(a[0] + D, b[0] + D) - lib.minimum(a[0], b[0]), 0)
    u = u_dx * u_dy

    return i / (epsilon + u)


def iou_accuracy(data1, data2):
    ious = [iou(tuple(data1[d][1]), tuple(data2[d][1])) for d in range(len(data1))]
    correct_bbs = np.greater_equal(ious, 0.5)

    correct_classes = [data1[d][0] == data2[d][0] for d in range(len(data1))]

    correct_predictions = np.logical_and(correct_bbs, correct_classes)

    return np.mean(correct_predictions)


def accuracy(data1, data2):
    data1_classes = np.array([x[0] for x in data1])
    data2_classes = np.array([x[0] for x in data2])
    return accuracy_score(data1_classes, data2_classes)


def mean_average_precision(y_true, y_pred):
    t_landmark_cls, t_landmark_reg, t_cls, t_reg = y_true
    p_landmark_cls, p_landmark_reg, p_cls, p_reg = y_pred

    s_cls = p_landmark_cls.shape
    s_reg = p_landmark_reg.shape

    # joining width and height dimensions into a single spacial dimension
    t_landmark_cls = np.reshape(t_landmark_cls, (s_cls[0], s_cls[1] * s_cls[2], s_cls[3]))
    t_landmark_reg = np.reshape(t_landmark_reg, (s_reg[0], s_reg[1] * s_reg[2], s_reg[3]))
    p_landmark_cls = np.reshape(p_landmark_cls, (s_cls[0], s_cls[1] * s_cls[2], s_cls[3]))
    p_landmark_reg = np.reshape(p_landmark_reg, (s_reg[0], s_reg[1] * s_reg[2], s_reg[3]))

    s_cls = p_landmark_cls.shape

    n_proposals = s_cls[1]
    n_classes = s_cls[2]

    # sorting by decreasing confidence
    pred_cls_confidences = np.max(p_landmark_cls, axis=-1)

    # tresholding all predictions with confidence < 0.5 to background.
    # background = np.zeros((p_landmark_cls.shape[2]))
    # background[0] = 1
    # p_landmark_cls[pred_cls_confidences < 0.5] = background

    pred_cls_sorted_confidences = np.sort(-pred_cls_confidences, axis=1)
    pred_cls_sorter = np.argsort(-pred_cls_confidences, axis=1)

    # I dont like this block, it should be possible to do this with fewer/more efficient operations.
    p_landmark_cls_sorted = []
    p_landmark_reg_sorted = []
    for i in range(p_landmark_cls.shape[0]):
        p_landmark_cls_sorted.append(p_landmark_cls[i, pred_cls_sorter[i]])
        p_landmark_reg_sorted.append(p_landmark_reg[i, pred_cls_sorter[i]])
    p_landmark_cls_sorted = np.array(p_landmark_cls_sorted)
    p_landmark_reg_sorted = np.array(p_landmark_reg_sorted)

    # combining cls and reg tensors.
    pred_cls_reg = np.concatenate((np.argmax(p_landmark_cls_sorted, axis=2)[:, :, np.newaxis], p_landmark_reg_sorted),
                                  axis=2)
    true_cls_reg = np.concatenate((np.argmax(t_landmark_cls, axis=2)[:, :, np.newaxis], t_landmark_reg), axis=2)

    # Lets calculate the precision recall curve.
    precision_recall_curve = np.full([n_classes, n_proposals, 2], None)
    for rank in range(n_proposals):

        y_true_ = true_cls_reg
        y_pred_rank = pred_cls_reg[:, slice(0, rank + 1), :]

        s_true = y_true_.shape
        s_pred = y_pred_rank.shape


        # inner joins predictions ground truths, and calculates IoUs
        y_pred_r = np.repeat(y_pred_rank, repeats=s_true[1], axis=1)
        y_true_r = np.tile(y_true_, [1, s_pred[1], 1])
        ious = np.transpose(tensor_iou(np, np.transpose(y_true_r[:, :, 1:]), np.transpose(y_pred_r[:, :, 1:])))

        # sets condition IoU > 0.5 and predicted label == ground-truth label
        tops_cond = np.logical_and(y_true_r[:, :, 0] == y_pred_r[:, :, 0], ious > iou_threshold)

        # (re-)merges setting  a true positive if any bb-gt pair respects the established conditions
        is_tp = np.any(np.reshape(tops_cond, [s_pred[0], s_pred[1], -1]), axis=2)

        for c in range(n_classes):

            tp_per_class = np.logical_and(y_pred_rank[:, :, 0] == c, is_tp)

            # any before sum, to count only up two one true positive per image, per class.
            tp_per_class = np.sum(np.any(tp_per_class, axis=-1))

            n_retrieved = np.sum(y_pred_rank[:, :, 0] == c)
            n_relevant = np.sum(y_true_[:, :, 0] == c)

            # skip aps
            if n_retrieved == 0 or n_relevant == 0:
                continue

            # print((tp_per_class / n_relevant))
            precision = float(tp_per_class) / n_retrieved
            recall = float(tp_per_class) / n_relevant
            precision_recall_curve[c, rank, :] = [precision, recall]

    aps = np.full([n_classes], None)

    for c in range(n_classes):
        ap_sum = 0.0
        for r in np.arange(0, 1.1, 0.1):
            try:
                # reading the precision values,
                precisions = precision_recall_curve[c, precision_recall_curve[c, :, 0] >= r, 1]
                p_interp = 0 if precisions.shape[0] == 0 else np.max(precisions)
            except:
                continue
            # print(p_interp)
            ap_sum += p_interp
        aps[c] = ap_sum / 11.0

    return aps



def main():
    """ Used for 'unit testing'
    """

    print('### TEST IOU ###')
    # a = np.array([
    #     [0, 1, 2, 3, 4],
    #     [0, 1, 2, 3, 4],
    # ])
    #
    # b = np.array([
    #     [1, 2, 3, 4, 5],
    #     [1, 2, 3, 4, 5],
    # ])
    #
    # print(a.shape)
    # print(b.shape)
    #
    # iou1 = iou(np, a, b)
    # iou2 = iou(np, b, a)
    # iou3 = iou(np, a, a)
    #
    # # res = sess.run([iou1, iou2, iou3])
    # print('Should be element wise equal')
    # print(iou1)
    # print(iou2)
    # print('Should be ~ 1')
    # print(iou3)
    #
    # print ("\n\n")

    print("### TEST mAP")

    img_size = (224, 224)  # n
    am_size = (12, 12)  #

    bb_size = 26  # 29
    stride = 18  # 13

    LOW_T = 0.3
    HIGH_T = 0.7

    dataset = ClopemaLoader()
    val_data = dataset.fetch_data('val')

    loader = LossLoader(dataset.n_landmark_cats(),
                        dataset.n_garment_cats(),
                        (12, 12),
                        26,
                        18)

    val = loader.as_input(val_data)
    # val[0][0, 0, 0, 5] = 10
    # val[0][0, 0, 1, 4] = 20
    # val[0][0, 8, 10, 3] = 25

    val = val[0][:, :, :-1], val[1][:, :, :-1], val[2], val[3]
    # print(val[0].shape)

    # val[0] =
    aps = mean_average_precision(val, val)
    print('mean average precision: ' + str(np.mean(aps[1:])))


if __name__ == '__main__':
    main()
