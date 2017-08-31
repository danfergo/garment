from glld.callbacks.history_checkpoint import load_losses
from glld.modules.simple_net import SimpleNet
from glld.modules.trid_net import TridNet
from glld.datasets.clopema import ClopemaLoader
from glld.util.display import Display
from keras import backend as K

from glld.util.util import accuracy, iou_accuracy


def main():
    with K.get_session():
        """ Used for testing'
        """

        train_name = 'with_bridge'
        use_bridge = True

        dataset = ClopemaLoader()
        m = SimpleNet(dataset.image_shape(), dataset.n_garment_cats(), dataset.n_landmark_cats(), use_bridge)
        display = Display(dataset, m.bb_size)

        data = dataset.fetch_data('val')
        data_x, data_y = data

        predictions = m.predict_combined(data_x, train_name)
        pred = m.loader.as_output(tuple(predictions))
        gt_loss = m.loader.as_input(tuple(data))

        display.show_landmark_stats(dataset.landmark_names, predictions, gt_loss)
        display.show_garment_stats(pred, data_y)


        display.history_charts(load_losses(train_name))
        display.show_multiple_results(data_x, pred,
                                      annotations=data_y,
                                      rpn_ground_truths=m.get_rpn_ground_truths(data))


if __name__ == '__main__':
    main()
