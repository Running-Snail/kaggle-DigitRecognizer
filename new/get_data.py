import pandas as pd
import numpy as np
import logging
import handle_data

format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=format
)

logging.info('loading data')
rawdata = pd.read_csv('../data/train.csv')

logging.info('spliting data')
mask = np.random.rand(len(rawdata)) < 0.8
train_set = rawdata[mask]
validate_set = rawdata[~mask]
logging.info('train set length ' + str(train_set.shape[0]))
logging.info('validate set length ' + str(validate_set.shape[0]))

logging.info('formatting data')
train_set_x = [r[2:] for r in train_set.itertuples()]
train_set_label = [r[1] for r in train_set.itertuples()]
train_set_y = [handle_data.y_for_label(l) for l in train_set_label]

validate_set_x = [r[2:] for r in validate_set.itertuples()]
validate_set_label = [r[1] for r in validate_set.itertuples()]
validate_set_y = [handle_data.y_for_label(l) for l in validate_set_label]
