from glld.callbacks.history_checkpoint import load_losses
from glld.modules.tree_net import TreeNet
from glld.modules.trid_net import TridNet
from glld.datasets.clopema import ClopemaLoader, filter_by_n_landmarks, filter_class, augment
from glld.util.display import Display
from keras import backend as K


def main():
    with K.get_session():
        """ Used for testing'
        """
        dataset = ClopemaLoader()
        data_x, data_y = augment(filter_by_n_landmarks(filter_class(dataset.fetch_data('val'), 0)))

        m = TreeNet((224, 224, 3), dataset.n_garment_cats(), dataset.n_landmark_cats())
        plot = Display(dataset, m.bb_size)
        m.load_weights(m.pants_detector, 'pants_overfit_again_night_1')

        # print(m[0].shape)
        pred = m.predict_pants_rois(data_x)
        # plot.history_charts(load_losses('pants_overfit_plz_3'))

        # ground_truths = m.get_rpn_ground_truths(data)
        plot.show_multiple_results(data_x, predictions=pred, annotations=data_y)


if __name__ == '__main__':
    main()



# input_layer = keras.layers.Input(shape=(224, 224, 3))
# feature_extractor = feature_extractor_model(input_layer)
#
# global_localizer_ = global_localizer(feature_extractor[1])
#
# model = keras.models.Model(
#     inputs=[input_layer],
#     outputs=global_localizer_
# )
#


# test(20, model)
