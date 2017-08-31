import keras.backend as k
import tensorflow as tf

epsilon = 1e-8


def robust_loss(x):
    x_abs = k.abs(x)
    x_bool = k.cast(k.less_equal(x_abs, 1.0), 'float32')
    return x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)


def rpn_cls_loss(y_true, y_pred):
    return k.binary_crossentropy(y_pred[:, :, :, :1], y_true[:, :, :, :1])


def spacial_loss(y_true, y_pred):
    i, h, w, c = tuple(y_pred.get_shape().as_list())

    # remove bg slice, reshape and permute dimensions y_true
    y_true_no_bg = y_true[:, :, :, 1:]
    y_true_no_bg = tf.reshape(y_true_no_bg, [-1, h * w, c - 1])
    y_true_no_bg = k.permute_dimensions(y_true_no_bg, (0, 2, 1))

    # remove bg slice, reshape and permute dimensions y_true
    y_pred_no_bg = y_pred[:, :, :, 1:]
    y_pred_no_bg = tf.reshape(y_pred_no_bg, [-1, h * w, c - 1])
    y_pred_no_bg = k.permute_dimensions(y_pred_no_bg, (0, 2, 1))
    y_pred_no_bg = k.softmax(y_pred_no_bg)

    # apply softmax, followed by spacial wise cross entropy and mask
    mask = tf.cast(tf.argmax(y_true_no_bg, axis=-1), 'float32')

    loss = mask * k.categorical_crossentropy(y_pred_no_bg, y_true_no_bg)

    # reverse, expand, normalize and add background slice
    loss = k.repeat(loss, h * w)
    bg_slice = loss[:, :, 0:1] * 0
    # print('-------------')
    # print(bg_slice.get_shape().as_list())
    # loss_sum = tf.reduce_sum(loss, axis=1, keep_dims=True)
    # normalized_loss = loss / (loss_sum + epsilon)
    complete_loss = tf.concat([bg_slice, loss], axis=-1)
    return tf.reshape(complete_loss, [-1, h, w, c])


def cls_loss(y_true, y_pred):
    is_active = y_true[:, :, :, -1]
    y_true_cls = y_true[:, :, :, :-1]
    return is_active * k.categorical_crossentropy(y_pred, y_true_cls)

    # i, h, w, c = tuple(y_pred.get_shape().as_list())

    # constraint_loss = tf.reshape(spacial_constraint(y_true, y_pred), [-1, 1, 1, c - 1])
    # constraint_loss = tf.tile(constraint_loss, [1, h, w, 1])
    # print(backgrounds.get_shape().as_list())
    # constraint_loss = tf.concat([backgrounds, constraint_loss], axis=-1)

    # print(constraint_loss.get_shape().as_list())
    # cls_loss = tf.expand_dims(cls_loss, -1)
    # cls_loss = tf.tile(cls_loss, [1, 1, 1, c - 1])
    # print('-x')
    # print(constraint_loss.get_shape().as_list())
    # constraint_loss +

    # print(is_active.get_shape().as_list())
    # print(y_true_.get_shape().as_list())
    # print(k.categorical_crossentropy(y_true_, y_pred).get_shape().as_list())
    # cls_loss = is_active * k.categorical_crossentropy(y_pred, y_true_)

    # constraint_reg = 0

    # return constraint_reg + cls_loss


def double_softmax(y_true, y_pred=None):
    i, h, w, c = tuple(y_pred.get_shape().as_list())
    cls_loss_value = cls_loss(y_true, k.softmax(y_pred))
    cls_loss_value = tf.reshape(cls_loss_value, [-1, h * w])
    cls_loss_value = tf.tile(cls_loss_value, (1, c))
    cls_loss_value = tf.reshape(cls_loss_value, [-1, h * w, c])
    cls_loss_value = tf.reshape(cls_loss_value, [-1, h, w, c])
    # loss_sum = tf.reduce_sum(cls_loss_value, axis=-1, keep_dims=True)
    # normalized_cls_loss = cls_loss_value / (loss_sum + epsilon)
    # normalized_cls_loss = tf.reshape(normalized_cls_loss, [-1, h, w, c])

    normalized_spacial_loss = spacial_loss(y_true[:, :, :, :-1], y_pred)
    return ((0.5 * cls_loss_value) + (0.5 * normalized_spacial_loss)) / 100


def rpn_reg_loss(y_true, y_pred):
    # print(y_true.shape)
    # print(y_pred.shape)
    y_true_reg = y_true[:, :, :, :2]
    y_true_cls = y_true[:, :, :, 2:]

    loss = robust_loss(y_pred - y_true_reg)

    return y_true_cls * loss


def roi_cls_loss(y_true, y_pred):
    return k.categorical_crossentropy(y_pred[:, :, :], y_true[:, :, :])


def roi_reg_loss(y_true, y_pred):
    # print(y_true.shape)
    y_true_reg = y_true[:, :, :2]
    y_true_cls = y_true[:, :, 2:]
    return y_true_cls * robust_loss(y_true_reg - y_pred)


def none_loss(y_true, y_pred):
    return tf.constant(0, tf.float32)
