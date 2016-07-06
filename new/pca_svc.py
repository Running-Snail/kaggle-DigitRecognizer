import pandas as pd
import numpy as np
import handle_data
from sklearn import cross_validation
from sklearn import metrics
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import logging
import matplotlib.pyplot as plt

format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=format
)
logging.info('loading data')
rawdata = pd.read_csv('../data/train.csv')

X = [r[2:] for r in rawdata.itertuples()]
y = [r[1] for r in rawdata.itertuples()]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state=42)

logging.info('fitting model')

scalar = preprocessing.MinMaxScaler()
svc = SVC(
    random_state=42,
    verbose=2
)
clf = Pipeline([('scale', scalar), ('pca', PCA(n_components=30)), ('model', svc)])
clf.fit(X_train, y_train)

# export model
joblib.dump(clf, 'model.pkl')

logging.info('model metrics')
y_predict = clf.predict(X_test)

mean_squared_error = metrics.mean_squared_error(y_test, y_predict)
logging.info('mean squared error {}'.format(mean_squared_error))
accuracy_score = metrics.accuracy_score(y_test, y_predict)
logging.info('accuracy score {}'.format(accuracy_score))
macro_precision_score = metrics.precision_score(y_test, y_predict, average='macro')
logging.info('macro_precision_score {}'.format(macro_precision_score))
micro_recall_score = metrics.precision_score(y_test, y_predict, average='micro')
logging.info('micro_recall_score {}'.format(micro_recall_score))
f1_score = metrics.f1_score(y_test, y_predict, average='weighted')
logging.info('f1 score {}'.format(f1_score))

for i in np.random.choice(range(len(y_predict)), 3, replace=False):
    row = X_test[i]
    img = handle_data.row_to_gray_img(row)
    label = y_test[i]
    plabel = y_predict[i]
    plt.xlabel('label {} predict {}'.format(label, plabel))
    plt.imshow(img)
    plt.show()

logging.info('generating test set result')
testdata = pd.read_csv('../data/test.csv')

X_validate = [row[1:] for row in testdata.itertuples()]

validate_y_predict = clf.predict(X_validate)

logging.info('showing some test set predicts')
for i in np.random.choice(range(len(validate_y_predict)), 10, replace=False):
    row = testdata.iloc[i]
    img = handle_data.row_to_gray_img(row)
    label = validate_y_predict[i]
    plt.xlabel('predict {}'.format(label))
    plt.imshow(img)
    plt.show()

test_result_df = pd.DataFrame(validate_y_predict)
test_result_df.index += 1
test_result_df.to_csv(
    'result.csv',
    index_label=['ImageId', 'Label']
)
