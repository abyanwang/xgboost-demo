import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.datasets._samples_generator import make_classification
# X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
X, y = make_classification(n_samples=10000, n_features=20, n_redundant=0,
                             n_clusters_per_class=1, n_classes=2, flip_y=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train)
dtrain = xgb.DMatrix(X_train,y_train)
dtest = xgb.DMatrix(X_test,y_test)

param = {'max_depth':6, 'eta':0.5, 'verbosity':1, 'objective':'binary:logistic'}
raw_model = xgb.train(param, dtrain, num_boost_round=20)

from sklearn.metrics import accuracy_score
pred_train_raw = raw_model.predict(dtrain)
for i in range(len(pred_train_raw)):
    if pred_train_raw[i] > 0.5:
         pred_train_raw[i]=1
    else:
        pred_train_raw[i]=0               
print (accuracy_score(dtrain.get_label(), pred_train_raw))
