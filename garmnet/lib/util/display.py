import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

from lib.util.metrics import mean_average_precision, accuracy, iou_accuracy


def bgr255_to_rgba(bgr255, alpha=1):
    color = [float(bgr255[2]) / 255, float(bgr255[1]) / 255, float(bgr255[0]) / 255, alpha]
    for i in range(4):
        color[i] = color[i] if color[i] < 1 else 1
    return tuple(color)


class Display:
    def __init__(self, dataset, bb_size):
        self.dataset = dataset
        self.bb_size = bb_size

    def show_rpn(self, rx, data_x, y):
        # stride = self.dataset.stride
        bb_size = self.bb_size

        # RPN PREDICTIONS
        rx.imshow(data_x)

        for i in range(len(y[3])):
            xy = int(y[3][i][0] - bb_size / 2), int(y[3][i][1] - bb_size / 2)
            c = y[3][i][2]

            if c == 0:
                rect = patches.Rectangle(xy, bb_size, bb_size,
                                         linewidth=1,
                                         edgecolor='r', facecolor='none')
                rx.add_patch(rect)

        for i in range(len(y[3])):
            xy = int(y[3][i][0] - bb_size / 2), int(y[3][i][1] - bb_size / 2)
            c = y[3][i][2]

            if c != 0:
                rect = patches.Rectangle(xy, bb_size, bb_size,
                                         linewidth=1,
                                         edgecolor='g', facecolor='none')

                rx.add_patch(rect)
                # if round(rpn_cls[i, j, 0]) > 0 and round(rpn_cls[i, j, 1]) < 1:
                #         edgecolor = 'g'
                #         label = 'positive'
                #     elif round(rpn_cls[i, j, 0]) < 1 and round(rpn_cls[i, j, 1]) > 0:
                #         edgecolor = (1, 0, 0, 0.1)
                #         # label = 'negative'
                #     else:
                #         edgecolor = 'b'
                # edgecolor = (0, 0, 0, 0.5)
                # label = 'neutral'
                # print('not zero neither one')
                #
                # if label == 'neutral':
                #     x = stride * j
                #     y = stride * i
                # else:
                #     x = stride * j + rpn_reg[i, j, 0]
                #     y = stride * i + rpn_reg[i, j, 1]

    def show_img(self, rx, x, y, up):
        landmark_colors = self.dataset.landmark_colors
        landmark_names = self.dataset.landmark_names
        bb_size = self.bb_size
        categories = self.dataset.categories

        # global
        rx.imshow(x)
        # print(y[0])
        if y[0] is not None:
            if up:
                rx.set_title(categories[y[0]])
            else:
                rx.set_xlabel(categories[y[0]])

        if y[1] is not None:
            reg = y[1]
            rect = patches.Rectangle((int(reg[0]), int(reg[1])), reg[2], reg[3],
                                     linewidth=2,
                                     edgecolor='w',
                                     facecolor='none')
            rx.add_patch(rect)

        # landmarks
        handles = []
        handles_classes = []
        for i in range(len(y[2])):  # landmark

            c = y[2][i][2]
            # print(c)
            confidence = 1 if len(y[2][i]) < 4 else y[2][i][3]

            if c == 0 or confidence < 0.5:
                continue

            clr = bgr255_to_rgba(landmark_colors[c], alpha=confidence)

            xy = int(y[2][i][0] - bb_size / 2), int(y[2][i][1] - bb_size / 2)

            rect = patches.Rectangle(xy, bb_size, bb_size,
                                     linewidth=1,
                                     edgecolor=clr,
                                     facecolor='none')
            rx.add_patch(rect)
            if c > 0:
                if c not in handles_classes:
                    handles.append(mlines.Line2D([], [], color=clr, label=landmark_names[c]))
                handles_classes.append(c)
            else:
                # print(p_roi_reg.shape)
                blk = (0, 0, 0, 0.05)
                rect = patches.Rectangle(xy, bb_size, bb_size,
                                         linewidth=1,
                                         edgecolor=blk,
                                         facecolor='none')
            rx.add_patch(rect)

            # rax(0, n).legend(handles=handles)

    def show_multiple_results(self, x, predictions=None, annotations=None, rpn_ground_truths=None):
        for i in range(15):
            slc = slice(i * 6, (i + 1) * 6)
            x_ = x[slc] if x is not None else None
            predictions_ = predictions[slc] if predictions is not None else None
            annotations_ = annotations[slc] if annotations is not None else None
            ground_truths_ = rpn_ground_truths[slc] if rpn_ground_truths is not None else None
            self.show_results(x_, predictions_, annotations_, ground_truths_)

    def show_results(self, x, predictions=None, annotations=None, ground_truths=None):

        k = min(6, x.shape[0])
        fig, ax = plt.subplots(4, k)

        def rax(r, c):
            if k == 1:
                return ax[r]
            else:
                return ax[r, c]

        rax(0, 0).set_ylabel('Predictions')
        rax(1, 0).set_ylabel('RPN predictions')
        rax(2, 0).set_ylabel('Anchor boxes')
        rax(3, 0).set_ylabel('Ground truth')

        # predictions
        for n in range(k):
            if predictions is not None:
                self.show_img(rax(0, n), x[n], predictions[n], True)
                self.show_rpn(rax(1, n), x[n], predictions[n])
            if ground_truths is not None:
                self.show_rpn(rax(2, n), x[n], ground_truths[n])

            if annotations is not None:
                self.show_img(rax(3, n), x[n], annotations[n], False)
        plt.show()

    def history_charts(self, history):

        # preview
        fig, ax = plt.subplots(2, 4)

        ax[0, 0].set_title('RPN cls')
        if 'rpn_cls_loss' in history:
            ax[0, 0].plot(history['rpn_cls_loss'])
        if 'val_rpn_cls_loss' in history:
            ax[0, 0].plot(history['val_rpn_cls_loss'])
        ax[0, 0].set_ylabel('Loss')
        ax[0, 0].set_xlabel('Epoch')
        ax[0, 0].legend(['Train', 'Validation'], loc='upper right')

        ax[1, 0].set_title('RPN reg')
        if 'rpn_reg_loss' in history:
            ax[1, 0].plot(history['rpn_reg_loss'])
        if 'val_rpn_reg_loss' in history:
            ax[1, 0].plot(history['val_rpn_reg_loss'])
        ax[1, 0].set_ylabel('Loss')
        ax[1, 0].set_xlabel('Epoch')
        ax[1, 0].legend(['Train', 'Validation'], loc='upper right')

        ax[0, 1].set_title('ROI cls')
        if 'roi_cls_loss' in history:
            ax[0, 1].plot(history['roi_cls_loss'])
        if 'val_roi_cls_loss' in history:
            ax[0, 1].plot(history['val_roi_cls_loss'])
        ax[0, 1].set_ylabel('Loss')
        ax[0, 1].set_xlabel('Epoch')
        ax[0, 1].legend(['Train', 'Validation'], loc='upper right')

        ax[1, 1].set_title('ROI reg')
        if 'roi_reg_loss' in history:
            ax[1, 1].plot(history['roi_reg_loss'])
        if 'val_roi_reg_loss' in history:
            ax[1, 1].plot(history['val_roi_reg_loss'])
        ax[1, 1].set_ylabel('Loss')
        ax[1, 1].set_xlabel('Epoch')
        ax[1, 1].legend(['Train', 'Validation'], loc='upper right')

        # ax[0, 2].set_title('ROI mAP')
        # print(np.greater_equal(np.array(map_metric.aps_history), 1.0))
        # for ap in map_metric.aps_history:
        #     ax[0, 2].plot(ap)

        ax[0, 2].set_title('Loss/Val_loss')
        if 'loss' in history:
            ax[0, 2].plot(history['loss'])
        if 'val_loss' in history:
            ax[0, 2].plot(history['val_loss'])
        ax[0, 2].set_ylabel('Loss')
        ax[0, 2].set_xlabel('Epoch')
        ax[0, 2].legend(['Train', 'Validation'], loc='upper right')

        ax[0, 3].set_title('Cls')
        if 'cls_loss' in history:
            ax[0, 3].plot(history['cls_loss'])
        if 'val_cls_loss' in history:
            ax[0, 3].plot(history['val_cls_loss'])
        ax[0, 3].set_ylabel('Loss')
        ax[0, 3].set_xlabel('Epoch')
        ax[0, 3].legend(['Train', 'Validation'], loc='upper right')

        ax[1, 3].set_title('Reg')
        if 'reg_loss' in history:
            ax[1, 3].plot(history['reg_loss'])
        if 'val_reg_loss' in history:
            ax[1, 3].plot(history['val_reg_loss'])
        ax[1, 3].set_ylabel('Loss')
        ax[1, 3].set_xlabel('Epoch')
        ax[1, 3].legend(['Train', 'Validation'], loc='upper right')

        plt.show()

    def show_landmark_stats(self, landmark_names, predictions, ground_truths):
        # print('EVALUATE:')
        # ground_truths = self.loader.anno_to_loss(ground_truths)
        # print(ground_truths[0].shape)
        ground_truths = \
            ground_truths[0][:, :, :, :-1], \
            ground_truths[1][:, :, :, :-1], \
            ground_truths[2], ground_truths[3]

        aps = mean_average_precision(ground_truths, tuple(predictions[0:2]) + (None, None))
        print(len(aps))
        for ap in range(1, len(aps)):
            print(landmark_names[ap] + ': ' + str(aps[ap]))
        # print(aps)
        print('MAP: ' + str(np.mean(aps[1:])))
        pass

    def show_garment_stats(self, pred, data_y):
        print('Classification accuracy: ' + str(round(accuracy(pred, data_y) * 100, 2)) + '%')
        print('Classification + localization accuracy: ' + str(round(iou_accuracy(pred, data_y) * 100, 2)) + '%')
