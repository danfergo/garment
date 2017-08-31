from glld.callbacks.history_checkpoint import load_losses
from glld.modules.trid_net import TridNet
from glld.datasets.clopema import ClopemaLoader
from glld.util.display import Display
from keras import backend as K


def main():
    with K.get_session():
        """ Used for testing'
        """
        dataset = ClopemaLoader()
        m = TridNet((224, 224, 3), dataset.n_garment_cats(), dataset.n_landmark_cats())
        display = Display(dataset)

        data = dataset.fetch_data('train')
        pred = m.predict_rpns(data[0], 'train_rpn')
        display.history_charts(load_losses('train_rpn'))
        display.show_multiple_results(data[0], pred,
                                      annotations=data[1],
                                      rpn_ground_truths=m.get_rpn_ground_truths(data))


if __name__ == '__main__':
    main()
