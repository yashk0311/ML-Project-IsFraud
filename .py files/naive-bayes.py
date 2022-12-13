import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer

train_df = pd.read_csv('../input/latest/train_df.csv')
y_train = pd.read_csv('../input/latest/y_train.csv')
test_df = pd.read_csv('../input/latest/test_df.csv')

y_train.drop(y_train.columns[[0]], axis=1, inplace=True)
test_df.drop(test_df.columns[[0]], axis=1, inplace=True)
train_df.drop(train_df.columns[[0]], axis=1, inplace=True)

params_NB = {'var_smoothing':np.logspace(0,-9, num=100)}

model = GaussianNB()
gs_NB = GridSearchCV(model, param_grid=params_NB, verbose=1,scoring='accuracy')
Data_transformed = PowerTransformer().fit_transform(train_df)
gs_NB.fit(Data_transformed, y_train);

y_pred = gs_NB.predict(test_df)

unique, counts = np.unique(y_pred, return_counts=True)

dict(zip(unique, counts))

pd.DataFrame(y_pred).to_csv('y_predNB_HPT1.csv')

gs_NB.best_params_
