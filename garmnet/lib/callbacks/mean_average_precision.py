from keras.callbacks import Callback
import numpy as np

class MAPCallback(Callback):
    aps_history = []
    map_history = []

    # def __init__(self):
    # super(MAPCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        y_true = self.validation_data[1:]
        y_pred = self.model.predict_global(self.validation_data[0], batch_size=1)

        n_proposals = y_true[2].shape[1]
        n_classes = y_true[2].shape[2]

        pred_cls_reg = np.concatenate((np.argmax(y_pred[2], axis=2)[:, :, np.newaxis], y_pred[3][:, :, :2]), axis=2)
        true_cls_reg = np.concatenate((np.argmax(y_true[2], axis=2)[:, :, np.newaxis], y_true[3][:, :, :2]), axis=2)

        # print('y_pred shape:' + str(y_pred.shape))

        # Lets calculate the precision recall curve.
        precision_recall_curve = np.full([n_classes, n_proposals, 2], None)
        for rank in range(n_proposals):
            y_true_ = true_cls_reg
            y_pred_ = pred_cls_reg[:, slice(0, rank + 1), :]

            s_pred = y_pred_.shape
            s_true = y_true_.shape

            y_pred_r = np.repeat(y_pred_, repeats=s_true[1], axis=1)
            y_true_r = np.tile(y_true_, [1, s_pred[1], 1])

            # inner joins predictions ground truths, and calculates IoUs
            IoUs = np.transpose(iou(np, np.transpose(y_true_r[:, :, 1:]), np.transpose(y_pred_r[:, :, 1:])))

            # sets condition IoU > 0.5 and predicted label == ground-truth label
            tops_cond = np.logical_and(y_true_r[:, :, 0] == y_pred_r[:, :, 0], IoUs > 0.5)

            # (re-)merges setting  a true positive if any bb-gt pair respects the established conditions
            is_tp = np.any(np.reshape(tops_cond, [s_pred[0], s_pred[1], -1]), axis=2)

            for c in range(n_classes):
                tp_per_class = y_pred_[np.logical_and(y_pred_[:, :, 1] == c, is_tp)].shape[0]
                # pred_per_class = y_pred_[y_pred_[:, :, 1] == c]
                # true_per_class = y_true_[y_true_[:, :, 1] == c]

                # print('.')

                # print(pred_per_class.shape)
                # print(true_per_class.shape)
                # print(pred_per_class.size)
                # print(true_per_class.size)
                # print(np.sum(y_true_[:, :, 1] == c))
                # print(np.sum(y_true_[:, :, 1] == c))
                tp_per_class = np.sum(np.logical_and(y_pred_[:, :, 1] == c, is_tp))
                n_retrieved = np.sum(y_pred_[:, :, 1] == c)
                n_relevant = np.sum(y_true_[:, :, 1] == c)

                # print(c)
                # print(rank)
                # print(pred_per_class.shape)
                # print(true_per_class.shape)
                # print(precision_recall_curve[c, rank].shape)
                if n_retrieved == 0 or n_relevant == 0:
                    continue
                    # else:
                    # print('has value')

                precision = tp_per_class / n_retrieved
                recall = tp_per_class / n_relevant

                precision_recall_curve[c, rank, :] = [precision, recall]
                # print('---')
                # print(pred_per_class.shape)
                # print(true_per_class.shape)

        for c in range(n_classes):
            ap_sum = 0.0
            for r in np.arange(0, 1.1, 0.1):
                try:
                    precisions = precision_recall_curve[c, precision_recall_curve[c, :, 0] > r, 1]
                    p_interp = 0 if precisions.shape[0] == 0 else np.max(precisions)
                except:
                    continue
                # print(p_interp)
                ap_sum += p_interp
            ap = ap_sum / 11.0

            if c >= len(self.aps_history):
                self.aps_history.append([])
            self.aps_history[c].append(ap)
