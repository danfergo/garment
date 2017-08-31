from keras import backend as k

from model.ground_truth_io import LossLoader
from lib.datasets.clopema import ClopemaLoader, augment
from model.garmnet import GarmNet
from lib.util.display import Display


ONE_SOFTMAX = 1
TWO_SOFTMAX_TRAIN = 0

def main():
    with k.get_session():
        """ Used for testing'
        """

        # CONFIGS
        include_bridge = True
        landmark_detector_softmax = ONE_SOFTMAX

        use_balanced_data = True
        augment_data = True
        n_epochs = 40
        n_trains = 1

        # (CONSTANT) CONFIGS
        balance = [1, 2, 5, 3, 3, 5, 6, 8, 8]

        dataset = ClopemaLoader()

        for i in range(n_trains):
            train_data = dataset.fetch_balanced_data('train', balance if use_balanced_data else None)
            train_data = augment(train_data) if augment_data else train_data

            val_data = dataset.fetch_data('val')

            model = GarmNet(dataset.image_shape(), dataset.n_garment_cats(), dataset.n_landmark_cats(), include_bridge, landmark_detector_softmax)
            loader = LossLoader(dataset.n_garment_cats(), dataset.n_landmark_cats(), model.get_bboxes_config())

            # model.train_landmarks('landmarks_without_spacial_constraint',
            #                       (train_data[0], loader.as_input(train_data)),
            #                       # (val_data[0], loader.as_input(val_data)),
            #                       (val_data[0], loader.as_input(val_data)),
            #                       n_epochs=n_epochs)

            model.train_combined(None, None, 'final_2',
                                  (train_data[0], loader.as_input(train_data)),
                                  # (val_data[0], loader.as_input(val_data)),
                                  (val_data[0], loader.as_input(val_data)),
                                  n_epochs=n_epochs)


if __name__ == '__main__':
    main()
