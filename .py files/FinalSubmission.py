import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.simplefilter("ignore")

train_df = pd.read_csv('../input/latest/train_df.csv')
y_train = pd.read_csv('../input/latest/y_train.csv')
test_df = pd.read_csv('../input/latest/test_df.csv')

y_train.drop(y_train.columns[[0]], axis=1, inplace=True)
test_df.drop(test_df.columns[[0]], axis=1, inplace=True)
train_df.drop(train_df.columns[[0]], axis=1, inplace=True)

estimator = xgb.XGBClassifier(colsample_bylevel=1,
              colsample_bytree=0.8,
              gpu_id=-1, learning_rate=0.02,
              max_depth=14,
              n_estimators=800,
              random_state=0,
              subsample=0.5, tree_method='gpu_hist')

est = estimator.fit(train_df,y_train)

y_pred = est.predict(test_df)
pd.DataFrame(y_pred).to_csv('y_predXGB_HPT.csv')