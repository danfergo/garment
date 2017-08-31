import numpy as np
import keras.backend as k


# def union(au, bu):
#     x = min(au[0], bu[0])
#     y = min(au[1], bu[1])
#     w = max(au[2], bu[2]) - x
#     h = max(au[3], bu[3]) - y
#     return x, y, w, h


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y

    # invalid intersection
    if w < 0 or h < 0:
        return 0, 0, 0, 0

    return x, y, w, h


def iou(a, b):
    # a nd b should be (x1,y1,x2,y2)

    # valid coordinates i.e., positive widths and heights.
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    i = intersection(a, b)
    # u = union(a, b)

    area_i = i[2] * i[3]
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    area_u = area_a + area_b - area_i

    # area_u = u[2] * u[3]
    return float(area_i) / float(area_u)


def coord_to_box(xy, size):
    x = xy[0]
    y = xy[1]
    return x, y, x + size, y + size


