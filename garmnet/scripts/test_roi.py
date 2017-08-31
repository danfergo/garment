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
        m.build(True)
        m.compile()
        m.load_weights(m.landmark_detector, 'train_roi')
        data = dataset.fetch_data('val')
        plot = Display(dataset)

        # print(m[0].shape)
        pred = m.predict_rois(data[0])
        ground_truths = m.get_rpn_ground_truths(data)
        for i in range(15):
            slc = slice(i * 6, (i + 1) * 6)
            plot.show_results(data[0][slc], pred[slc], annotations=data[1][slc], ground_truths=ground_truths[slc])
            return
        # print(len(data))
        # print('-')


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
