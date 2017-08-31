from glld.callbacks.history_checkpoint import load_losses
from glld.modules.simple_net import SimpleNet
from glld.modules.trid_net import TridNet
from glld.datasets.clopema import ClopemaLoader
from glld.util.display import Display
from keras import backend as K


def main():
    with K.get_session():
        """ Used for testing'
        """
        dataset = ClopemaLoader()
        m = SimpleNet(dataset.image_shape(), dataset.n_garment_cats(), dataset.n_landmark_cats())
        display = Display(dataset, m.bb_size)

        data = dataset.fetch_data('val')

        predictions = m.predict_landmarks(data[0], 'landmarks_with_spacial_constraint', ground_truths=data)
        pred = m.loader.as_output(tuple(predictions) + (None, None))
        gt_loss = m.loader.as_input(tuple(data))

        display.show_landmark_stats(dataset.landmark_names, predictions, gt_loss)

        display.history_charts(load_losses('landmarks_with_spacial_constraint'))
        display.show_multiple_results(data[0], pred,
                                      annotations=data[1],
                                      rpn_ground_truths=m.get_rpn_ground_truths(data))


if __name__ == '__main__':
    main()
