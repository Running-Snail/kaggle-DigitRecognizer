import pandas as pd
import numpy as np

def label_for_row(row):
    return row[1]

def y_for_label(label, n=10):
    ret = [0]*n
    ret[label-1] = 1
    return ret

def label_for_y(y, n=10):
    return np.argmax(y)+1

def pixels_for_row(row):
    return row[2:]

def row_to_gray_img(row):
    d3 = np.array(map(lambda x: [x/256.0, x/256.0, x/256.0], row)).reshape((28, 28, 3))
    return d3
