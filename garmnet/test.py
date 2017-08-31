from keras import backend as k

from lib.callbacks.history_checkpoint import load_losses
from lib.datasets.clopema import ClopemaLoader, augment
from model.garmnet import GarmNet
from lib.util.display import Display
from model.ground_truth_io import LossLoader

ONE_SOFTMAX = 1
TWO_SOFTMAX_TEST = 2


def main():
    with k.get_session():
        """ Used for testing'
        """

        # CONFIGS
        include_bridge = True
        landmark_detector_softmax = TWO_SOFTMAX_TEST
        test_name = 'final_1'

        # (CONSTANT) CONFIGS
        dataset = ClopemaLoader()

        data = dataset.fetch_data('val')

        model = GarmNet(dataset.image_shape(),
                        dataset.n_garment_cats(),
                        dataset.n_landmark_cats(),
                        include_bridge,
                        landmark_detector_softmax)
        loader = LossLoader(dataset.n_garment_cats(),
                            dataset.n_landmark_cats(),
                            model.get_bboxes_config())

        plot = Display(dataset, model.bb_size)
        plot.history_charts(load_losses(test_name))


        # Predictions
        # pred = model.predict_landmarks(data[0], test_name, ground_truths=data)
        # predictions = loader.as_output(tuple(pred) + (None, None))
        #
        # pred = model.predict_global(data[0], test_name)
        # predictions = loader.as_output((None, None) + tuple(pred))

        pred = model.predict_combined(data[0], test_name)
        predictions = loader.as_output(tuple(pred))


        ground_truths = loader.as_input(data)
        ground_truths_processed = loader.as_output(ground_truths)


        plot.show_landmark_stats(dataset.landmark_names, pred, ground_truths)

        plot.show_garment_stats(predictions, data[1])
        #
        plot.show_multiple_results(data[0], predictions,
                                   annotations=data[1],
                                   rpn_ground_truths=ground_truths_processed)


if __name__ == '__main__':
    main()
