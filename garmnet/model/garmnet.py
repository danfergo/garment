import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
import time

from lib.callbacks.history_checkpoint import HistoryCheckpoint
from model.losses import rpn_reg_loss, cls_loss, double_softmax
from model.blocks import feature_extractor_model, rpn_model, garment_localizer
from keras.callbacks import ModelCheckpoint

import os

parent_path = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))


class GarmNet:
    """
        This three values were fixed so that the three match the net's input size (224, 224)
    """
    bb_size = 26
    am_size = (12, 12)
    stride = 18

    def __init__(self, input_shape, n_garment_cats, n_landmark_cats, include_bridge=False, landmark_detector_softmax=1):
        self.n_garment_cats = n_garment_cats
        self.n_landmark_cats = n_landmark_cats
        self.input_shape = input_shape

        print('N garment cats ' + str(n_garment_cats))
        print('N landmark cats ' + str(n_landmark_cats))

        self.build(include_bridge, landmark_detector_softmax)
        self.compile(landmark_detector_softmax != 1)

    def build(self, include_bridge, landmark_detector_softmax):
        """
        Joins the blocks into the three models: garment localizer, landmark detector and combined (GarmNet)
        :param include_bridge: (see the thesis)
        :return: void
        """

        input_layer = Input(shape=self.input_shape)
        feature_extractor = feature_extractor_model(input_layer)
        landmark_detector_head = rpn_model(feature_extractor[0], landmark_detector_softmax,
                                           n_classes=self.n_landmark_cats)

        if include_bridge:
            bridge_layer = Flatten()(landmark_detector_head[0])
            garmnet_localizer_head = garment_localizer(feature_extractor[1], bridge_layer)
        else:
            garmnet_localizer_head = garment_localizer(feature_extractor[1])

        self.garment_detector = Model(
            inputs=[input_layer],
            outputs=garmnet_localizer_head
        )

        self.landmarks_detector = Model(
            inputs=[input_layer],
            outputs=landmark_detector_head
        )

        self.garmnet = Model(
            inputs=[input_layer],
            outputs=landmark_detector_head + garmnet_localizer_head
        )

    def compile(self, use_double_softmax_loss):
        # optimizer=Adadelta(lr=0.5, rho=0.95, epsilon=1e-8, decay=0.),

        if use_double_softmax_loss:
            print('using double softmax loss function')

        self.garment_detector.compile(
            optimizer='adadelta',
            loss={
                'cls': 'categorical_crossentropy',
                'reg': 'mean_squared_error'
            })

        self.landmarks_detector.compile(
            optimizer='adadelta',
            loss={
                'rpn_cls': double_softmax if use_double_softmax_loss else cls_loss,
                'rpn_reg': rpn_reg_loss
            })

        self.garmnet.compile(
        # optimizer=Adadelta(lr=0.5, rho=0.95, epsilon=1e-8, decay=0.),

            optimizer='adadelta',
            loss={
                'cls': 'categorical_crossentropy',
                'reg': 'mean_squared_error',
                'rpn_cls': double_softmax if use_double_softmax_loss else cls_loss,
                'rpn_reg': rpn_reg_loss
            })

    def get_bboxes_config(self):
        return self.bb_size, self.stride, self.am_size

    #
    # def fetch_data(self, dataset, split_name):
    #     paths = dataset.get_paths(split_name)
    #     train_x = np.load(paths[0], encoding='bytes')
    #     train_y = np.load(paths[1], encoding='bytes')
    #     return self.loader.as_input((train_x, train_y))

    def load_weights(self, to, train_name):
        if train_name is not None:
            full_path = parent_path + '/history/' + train_name
            try:
                to.load_weights(full_path)
                print('Weights restored.')
            except Exception as e:
                print('Failed to restore weights. (' + full_path + ')')
                print('Error:' + str(e))

    def __predict(self, model, data, batch_size):
        return model.predict(
            data,
            batch_size=batch_size
        )

    def predict_global(self, data, train_name=None, batch_size=1):
        self.load_weights(self.garment_detector, train_name)
        return self.__predict(self.garment_detector, data, batch_size)

    def predict_combined(self, data, train_name=None, batch_size=1):
        self.load_weights(self.garmnet, train_name)

        start = time.time()
        predictions = self.__predict(self.garmnet, data, batch_size)
        end = time.time()

        elapsed = end - start
        print('FPS: ' + str(float(len(data[0]) / elapsed)))

        return predictions

    def predict_landmarks(self, data, train_name=None, batch_size=1, ground_truths=None):
        self.load_weights(self.landmarks_detector, train_name)
        return self.__predict(self.landmarks_detector, data, batch_size)

    def __train(self, model, train_name, train_data, val_data, data_slice, n_epochs, batch_size):
        train_x, train_y = train_data
        val_x, val_y = val_data

        # train = self.loader.anno_to_loss(train_data)
        # val = self.loader.anno_to_loss(val_data)

        # mean_average_precision(val, val)

        historyCheckpoint = HistoryCheckpoint(train_name)
        self.load_weights(model, train_name)

        # train_x
        model.fit(x=train_x, y=list(train_y[data_slice]),
                  # verbose=0,
                  callbacks=[
                      historyCheckpoint,
                      ModelCheckpoint(
                          parent_path + '/history/' + train_name,
                          save_weights_only=True
                      )
                  ],
                  validation_data=(val_x, list(val_y[data_slice])),
                  batch_size=batch_size,
                  epochs=n_epochs,
                  # initial_epoch=historyCheckpoint.n_past_epochs()
                  )
        return historyCheckpoint.re_save()

    def train_landmarks(self, train_name, train_data, val_data, n_epochs=1, batch_size=30):
        slc = slice(0, 2)
        history = self.__train(self.landmarks_detector, train_name, train_data, val_data, slc, n_epochs, batch_size)
        return history, self.predict_landmarks(val_data[0])

    def train_global(self, train_name, train_data, val_data, n_epochs=1, batch_size=30):
        slc = slice(2, 4)
        history = self.__train(self.garment_detector, train_name, train_data, val_data, slc, n_epochs, batch_size)
        return history, self.predict_global(val_data[0])

    def train_combined(self, train_name_global, train_name_landmarks, train_name, train_data, val_data, n_epochs,
                       batch_size=30):
        slc = slice(0, 4)

        if train_name_global is not None:
            self.load_weights(self.garment_detector, train_name_global)
        if train_name_landmarks is not None:
            self.load_weights(self.landmarks_detector, train_name_landmarks)

        history = self.__train(self.garmnet, train_name, train_data, val_data, slc, n_epochs, batch_size)
        return history, self.predict_global(val_data[0])
