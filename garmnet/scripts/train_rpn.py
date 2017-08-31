from keras import backend as k

from glld.datasets.clopema import ClopemaLoader
from glld.modules.trid_net import TridNet
from glld.util.display import Display


def main():
    with k.get_session():
        """ Used for testing'
        """
        dataset = ClopemaLoader()
        plot = Display(dataset)

        m = TridNet((224, 224, 3), dataset.n_garment_cats(), dataset.n_landmark_cats())

        train_data, val_data = dataset.fetch_multiple_data(['train', 'val'])
        m.get_rpn_ground_truths(val_data)

        results = m.train_rpn('train_rpn', train_data, val_data, n_epochs=1)

        plot.history_charts(results[0])
        plot.show_results(val_data[0], results[1], val_data[1], ground_truths=results[2])


if __name__ == '__main__':
    main()
