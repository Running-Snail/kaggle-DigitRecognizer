from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# get data
print('reading data')
rawdata = pd.read_csv('../../data/train.csv')
validate = rawdata[:8400]
train = rawdata[8400:]

print('validate set shape', validate.shape)
print('train set shape', train.shape)

def label_for_row(row):
    return row[1]

def pixels_for_row(row):
    return row[2:]

y = [label_for_row(r) for r in train.itertuples()]
x = [pixels_for_row(r) for r in train.itertuples()]
validata_y = [label_for_row(r) for r in validate.itertuples()]
validate_x = [pixels_for_row(r) for r in validate.itertuples()]

print('fitting data')
clf = RandomForestClassifier(n_estimators=25)
clf.fit(x, y)

print('using validation set')
predict_y = clf.predict(validate_x)

precision = 0.0
for i, py in enumerate(predict_y):
    if py == validata_y[i]:
        precision += 1
print('precision is', precision/len(y))
