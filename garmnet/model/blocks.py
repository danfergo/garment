from keras.applications import ResNet50
from keras.layers import Conv2D, BatchNormalization, Activation, Lambda, \
    Flatten, Dense, regularizers, Concatenate

import tensorflow as tf
import keras.backend as k


def feature_extractor_model(input_layer):
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
    # for layer in resnet.layers:
    #     print(layer.name)
    output_layer = resnet.get_layer('activation_40').output
    # output_layer2 = resnet.get_layer('activation_49').output
    # print('Feature extractor shapes')
    # print(output_layer.get_shape().as_list())
    # print(resnet.output.get_shape().as_list())
    return [output_layer, resnet.output]


def double_activation_map(x):
    epsilon = 1e-8
    i, h, w, c = tuple(x.get_shape().as_list())

    # CLS
    cls_loss_value = k.softmax(x)
    # cls_loss_value = tf.reshape(cls_loss_value, [-1, h * w, c])
    # cls_loss_value = k.repeat(cls_loss_value, c)
    # cls_loss_value = k.permute_dimensions(cls_loss_value, (0, 2, 1))
    # loss_sum = tf.reduce_sum(cls_loss_value, axis=-1, keep_dims=True)
    # normalized_cls_loss = cls_loss_value / (loss_sum + epsilon)
    # normalized_cls_loss = tf.reshape(normalized_cls_loss, [-1, h, w, c])

    # SPACIAL
    # remove bg slice, reshape and permute dimensions y_true
    y_pred_no_bg = x[:, :, :, 1:]
    y_pred_no_bg = tf.reshape(y_pred_no_bg, [-1, h * w, c - 1])
    y_pred_no_bg = k.permute_dimensions(y_pred_no_bg, (0, 2, 1))
    bg_slice = y_pred_no_bg[:, 0:1, :] * 0

    print(y_pred_no_bg.get_shape().as_list())
    y_pred_no_bg = k.softmax(y_pred_no_bg)
    # print(y_pred_no_bg.get_shape().as_list())

    spacial_loss = tf.concat([bg_slice, y_pred_no_bg], axis=1)
    # print(spacial_loss.get_shape().as_list())

    spacial_loss = k.permute_dimensions(spacial_loss, (0, 2, 1))
    spacial_loss = tf.reshape(spacial_loss, [-1, h, w, c])
    # print(spacial_loss.get_shape().as_list())


    # reverse, expand, normalize and add background slice
    # loss = k.repeat(y_pred_no_bg, h * w)
    # loss_sum = tf.reduce_sum(loss, axis=1, keep_dims=True)
    # normalized_loss = loss / (loss_sum + epsilon)

    # normalized_spacial_loss = tf.reshape(complete_loss, [-1, h, w, c])
    #
    return (0.5 * cls_loss_value) + (0.5 * spacial_loss)


def rpn_model(input_layer, landmark_detector_softmax, n_classes=2, n_intermediate=256):
    n_ratios = 1
    n_scales = 1
    n_boxes = n_ratios * n_scales

    intermediate_layer = Conv2D(n_intermediate, (3, 3),
                                activation='relu')(input_layer)

    # rpn cls head
    cls_head = Conv2D(n_classes * n_boxes, (1, 1),
                      kernel_initializer='random_normal')(intermediate_layer)

    cls_head = BatchNormalization()(cls_head)
    if landmark_detector_softmax == 0:
        cls_head = Lambda(lambda x: x, name='rpn_cls')(cls_head)
    elif landmark_detector_softmax == 1:
        cls_head = Activation(k.softmax, name='rpn_cls')(cls_head)
    elif landmark_detector_softmax == 2:
        cls_head = Lambda(double_activation_map, name='rpn_cls')(cls_head)

        # rpn reg head
    reg_head = Conv2D(2 * n_boxes, (1, 1),
                      kernel_initializer='random_normal',
                      bias_initializer='ones',
                      activation='relu', name='rpn_reg')(intermediate_layer)

    return [cls_head, reg_head]


# def landmark_localizer_model(input_layer, n_classes):
#     roi_layer_s = input_layer.get_shape().as_list()
#
#     # roi shared layers
#     reshape = Lambda(lambda x: k.reshape(x, tuple(roi_layer_s[1:])))(
#         input_layer)
#     roi_conv_layer_1 = Conv2D(256, (3, 3),
#                               padding='same',
#                               activation='relu')(reshape)
#     roi_conv_layer_1 = BatchNormalization()(roi_conv_layer_1)
#     roi_conv_layer_1 = Activation('relu')(roi_conv_layer_1)
#
#     roi_conv_layer_1 = Conv2D(256, (3, 3),
#                               padding='same',
#                               activation='relu')(roi_conv_layer_1)
#     roi_conv_layer_1 = BatchNormalization()(roi_conv_layer_1)
#     roi_conv_layer_1 = Activation('relu')(roi_conv_layer_1)
#
#     # roi cls head
#     roi_cls_layer = Flatten()(roi_conv_layer_1)
#     roi_cls_layer = Dense(n_classes)(roi_cls_layer)
#     # print(roi_cls_layer.get_shape().as_list())
#
#     roi_cls_layer = Lambda(lambda x: k.reshape(x, (1, roi_layer_s[0] * roi_layer_s[1], n_classes)))(roi_cls_layer)
#     roi_cls_layer = BatchNormalization()(roi_cls_layer)
#     roi_cls_layer = Activation('softmax', name='roi_cls')(roi_cls_layer)
#
#     # roi reg head
#     roi_reg_layer = Flatten()(roi_conv_layer_1)
#     roi_reg_layer = Dense(2)(roi_reg_layer)
#
#     roi_reg_layer = Lambda(lambda x: k.reshape(x, (1, roi_layer_s[0] * roi_layer_s[1], 2)), name='roi_reg')(
#         roi_reg_layer)
#
#     return [roi_cls_layer, roi_reg_layer]


def garment_localizer(input_layer, extra_layer=None):
    common_layer = Flatten()(input_layer)

    if extra_layer is not None:
        common_layer = Concatenate()([common_layer, extra_layer])

    common_layer = Dense(512)(common_layer)
    common_layer = BatchNormalization()(common_layer)
    common_layer = Activation('relu')(common_layer)

    cls_layer = Dense(9)(common_layer)
    cls_layer = BatchNormalization()(cls_layer)
    cls_layer = Activation('softmax', name='cls')(cls_layer)

    reg_layer = Dense(4, name='reg')(common_layer)

    return [cls_layer, reg_layer]


def class_specific_detector(input_layer, n_values):
    intermediate_layer = Flatten()(input_layer)

    intermediate_layer = Dense(512,
                               kernel_regularizer=regularizers.l2(0.01))(intermediate_layer)
    intermediate_layer = BatchNormalization()(intermediate_layer)
    intermediate_layer = Activation('relu')(intermediate_layer)

    reg_layer = Dense(n_values,
                      kernel_regularizer=regularizers.l2(0.01),
                      name='reg')(intermediate_layer)

    return [reg_layer]


def main():
    """ Used for 'unit testing'
    """

    # fast_rcnn()


if __name__ == '__main__':
    main()
