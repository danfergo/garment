from glld.modules.simple_net import SimpleNet
from glld.modules.trid_net import TridNet
from glld.datasets.clopema import ClopemaLoader
from glld.util.display import Display
from keras import backend as K
from glld.callbacks.history_checkpoint import load_losses

from glld.util.util import iou_accuracy, accuracy


def main():
    with K.get_session():
        """ Used for testing'
        """
        dataset = ClopemaLoader()

        m = SimpleNet(dataset.image_shape(), dataset.n_garment_cats(), dataset.n_landmark_cats(), False)
        display = Display(dataset, m.bb_size)

        data = dataset.fetch_data('val')
        data_x, data_y = data

        predictions = m.predict_global(data_x, 'global')
        pred = m.loader.as_output((None, None) + tuple(predictions))

        display.show_garment_stats(pred, data_y)

        display.history_charts(load_losses('global'))
        display.show_multiple_results(data_x, pred, data_y)


if __name__ == '__main__':
    main()
