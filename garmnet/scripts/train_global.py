import matplotlib.pyplot as plt

from keras import backend as K

from glld.datasets.clopema import ClopemaLoader, augment
from glld.modules.simple_net import SimpleNet
from glld.modules.trid_net import TridNet
from glld.util.display import Display
from glld.callbacks.history_checkpoint import load_losses


def main():
    with K.get_session():
        """ Used for testing'
        """
        dataset = ClopemaLoader()

        m = SimpleNet(dataset.image_shape(), dataset.n_garment_cats(), dataset.n_landmark_cats(), False)
        display = Display(dataset, m.bb_size)

        train_data, val_data = dataset.fetch_multiple_data(['train', 'val'])
        # train_data = augment(train_data)

        results = m.train_global('global', train_data, val_data, n_epochs=40)

        display.history_charts(results[0])
        # display.show_results(train_data[0], results[1], train_data[1])


if __name__ == '__main__':
    main()
