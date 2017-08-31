import matplotlib.pyplot as plt

from keras import backend as K
import numpy as np

from glld.datasets.clopema import ClopemaLoader, augment, filter_by_n_landmarks, filter_class
from glld.modules.tree_net import TreeNet
from glld.modules.trid_net import TridNet
from glld.util.display import Display


def main():
    with K.get_session():
        """ Used for testing'
        """

        dataset = ClopemaLoader()
        plot = Display(dataset)

        m = TreeNet((224, 224, 3), dataset.n_garment_cats(), dataset.n_landmark_cats())

        train_data, val_data = dataset.fetch_multiple_data(['train', 'val'])
        val_data = (filter_by_n_landmarks(filter_class(val_data, 0)))

        # plot.show_results(val_data[0], predictions=m.load_losses(val_data), annotations=val_data[1])
        train_data = augment(filter_by_n_landmarks(filter_class(train_data, 0)))

        results = m.train_roi_pants('pants_overfit_again_night_1', train_data, val_data, n_epochs=10)

        plot.history_charts(results[0])
        plot.show_results(val_data[0], predictions=results[1], annotations=val_data[1])


if __name__ == '__main__':
    main()
