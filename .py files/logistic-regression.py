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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('../input/latest/train_df.csv')
y_train = pd.read_csv('../input/latest/y_train.csv')
test_df = pd.read_csv('../input/latest/test_df.csv')

y_train.drop(y_train.columns[[0]], axis=1, inplace=True)
test_df.drop(test_df.columns[[0]], axis=1, inplace=True)
train_df.drop(train_df.columns[[0]], axis=1, inplace=True)

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(train_df,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

y_pred = logreg_cv.predict(test_df)

pd.DataFrame(y_pred).to_csv('y_predLRP.csv')

