from keras import backend as k

from glld.datasets.clopema import ClopemaLoader, augment
from glld.modules.simple_net import SimpleNet
from glld.util.display import Display


def main():
    with k.get_session():
        """ Used for testing'
        """
        dataset = ClopemaLoader()

        m = SimpleNet(dataset.image_shape(), dataset.n_garment_cats(), dataset.n_landmark_cats(), True)
        plot = Display(dataset, m.bb_size)

        # train_data, val_data = dataset.fetch_multiple_data(['train', 'val'])

        train_data = dataset.fetch_multiple_balanced_data(['train'])[0]
        val_data = dataset.fetch_data('val')
        # print(train_data[0].shape)

        ground_truths = m.get_rpn_ground_truths(val_data)

        results = m.train_combined('global', 'landmarks_without_spacial_constraint', 'with_bridge_2', train_data, val_data, n_epochs=10)

        # for i in range(10):
        #     train_data_ = train_data if i == 2 == 0 else augment(train_data)
        #     results = m.train_combined(None, None, 'with_bridge_augmented', train_data_, val_data, n_epochs=5)

        plot.history_charts(results[0])
        plot.show_results(val_data[0], results[1], val_data[1], ground_truths=ground_truths)


if __name__ == '__main__':
    main()
