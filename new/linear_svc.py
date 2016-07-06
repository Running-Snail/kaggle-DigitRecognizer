import pandas as pd
import numpy as np
import get_data
import handle_data
from sklearn import linear_model
from sklearn.svm import LinearSVC
import logging
import matplotlib.pyplot as plt

logging.info('scaling data')
scaled_train_set_x = (get_data.train_set_x - np.min(get_data.train_set_x)) / (np.max(get_data.train_set_x) + 0.0001)

logging.info('fitting model')

clf = LinearSVC()
clf.fit(scaled_train_set_x, get_data.train_set_label)

logging.info('model precision')
predict_label = clf.predict(get_data.validate_set_x)

precision_arr = np.array(get_data.validate_set_label) == np.array(predict_label)
precision = float(np.sum(precision_arr))/len(precision_arr)

for i in np.random.choice(range(len(predict_label)), 3, replace=False):
    print(i)
    row = get_data.validate_set.iloc[i]
    img = handle_data.row_to_gray_img(row[1:])
    label = get_data.validate_set_label[i]
    plabel = predict_label[i]
    plt.xlabel('label {} predict {}'.format(label, plabel))
    plt.imshow(img)
    plt.show()
logging.info('overall precision {}'.format(precision))

logging.info('generating test set result')
testdata = pd.read_csv('../data/test.csv')

test_set_x = [row[1:] for row in testdata.itertuples()]
test_predict_label = clf.predict(test_set_x)

logging.info('showing some test set predicts')
for i in np.random.choice(range(len(test_predict_label)), 10, replace=False):
    print(i)
    row = testdata.iloc[i]
    print(len(row))
    img = handle_data.row_to_gray_img(row)
    label = test_predict_label[i]
    plt.xlabel('predict {}'.format(label))
    plt.imshow(img)
    plt.show()

test_result_df = pd.DataFrame(test_predict_label)
test_result_df.index += 1
test_result_df.to_csv(
    'result.csv',
    index_label=['ImageId', 'Label']
)
