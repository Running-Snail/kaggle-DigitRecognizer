import numpy as np


def label_to_y(label, size=10):
    r = np.zeros(size)
    r[label] = 1.0
    return r


def handle_row_x(data, row):
    return (data.values[row][1:]-128.0)/256.0


def handle_row_y(data, row):
    return label_to_y(data.values[row][0])


def handle_data(data):
    rows = data.shape[0]
    x = np.array([handle_row_x(data, i) for i in xrange(rows)])
    y = np.array([handle_row_y(data, i) for i in xrange(rows)])
    return x, y


def row_to_gray_img(row):
    d3 = np.array(map(lambda x: [x/256.0, x/256.0, x/256.0], dat.values[row][1:])).reshape((28, 28, 3))
    return d3
