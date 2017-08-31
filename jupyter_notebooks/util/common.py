import numpy as np


def to_rgb_img(y, color_map, argmax=False):
    def map_avg_row_color(r):
        x = int(np.average(r))
        x = max(x, 0)
        x = min(58, x)
        return color_map[x]

    def map_argmax_row_color(r):
        x = int(np.argmax(r))
        x = max(x, 0)
        x = min(58, x)
        return color_map[x]

    s = y.shape
    yy = np.zeros((s[0], s[1], 3))
    if not argmax:
        yy = np.apply_along_axis(map_avg_row_color, 2, y)
    else:
        yy = np.apply_along_axis(map_argmax_row_color, 2, y)
    return yy


def to_categorical(y):
    def map_row(r):
        r[int(r[-1])] = 1
        return r

    yy = np.array(y)
    s = yy.shape
    yyy = np.expand_dims(yy, axis=3)
    tmp = np.zeros((s[0], s[1], s[2], 59))
    tmp[:, :, :, -1:] = yyy[:, :, :, :1]
    tmp = np.apply_along_axis(map_row, 3, tmp)[:, :, :, :-1]
    return tmp


def from_categorical(y):
    def map_row(r):
        r[int(r[-1])] = 1
        return r

    yy = np.array(y)
    s = yy.shape
    yyy = np.expand_dims(yy, axis=3)
    tmp = np.zeros((s[0], s[1], s[2], 59))
    tmp[:, :, :, -1:] = yyy[:, :, :, :1]
    tmp = np.apply_along_axis(np.max, 3, tmp)
    return tmp
