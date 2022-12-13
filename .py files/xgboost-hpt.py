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
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('../input/latest/train_df.csv')
y_train = pd.read_csv('../input/latest/y_train.csv')
test_df = pd.read_csv('../input/latest/test_df.csv')

y_train.drop(y_train.columns[[0]], axis=1, inplace=True)
test_df.drop(test_df.columns[[0]], axis=1, inplace=True)
train_df.drop(train_df.columns[[0]], axis=1, inplace=True)

grid_params = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

estimator = xgb.XGBClassifier()

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=grid_params,
    scoring = 'accuracy',
    n_jobs = 10,
    cv = 10,
    verbose=True
)

grid_search.fit(train_df,y_train)

y_pred = grid_search.predict(test_df)

pd.DataFrame(y_pred).to_csv('y_predXGB_HPT.csv')