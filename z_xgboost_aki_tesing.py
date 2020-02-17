'''
1. conda current xgboost
2. noncodna xgboost with all uniform dist
3. nonconda xgboost with specific p for each variable

A. Sample rows
B. all rows
'''

print("1 B")
from sklearn.datasets import load_boston
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import roc_auc_score,  roc_curve
from matplotlib import pyplot
from sklearn.metrics import auc
import seaborn as sns
from sklearn import metrics

print("xgb.__version__ : ",xgb.__version__)
#data_dir= '~/projects/AKI/AKI_test_data'
#data_dir= '~/projects/AKI/test'
data_dir='python-package/data/'
train_csv = os.path.join(data_dir,'train_csv.csv')
test_csv = os.path.join(data_dir,'test_csv.csv')
weight_csv = os.path.join(data_dir,'weight_csv.csv')

train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
weight = pd.read_csv(weight_csv)


interest_cols = ["2160.0","375983.01_cum","48642.3","48642.3_change","718.7","AGE","BMI","CH.71010","RACE_01","RACE_02","RACE_03","RACE_04","RACE_05","RACE_06","RACE_07","RACE_NI","RACE_OT","SEX_F","SEX_M","SEX_NI","y"]

X_col = ["2160.0","375983.01_cum","48642.3","48642.3_change","718.7","AGE","BMI","CH.71010","RACE_01","RACE_02","RACE_03","RACE_04","RACE_05","RACE_06","RACE_07","RACE_NI","RACE_OT","SEX_F","SEX_M","SEX_NI"]
y_col = ['y']



train = train[interest_cols]
test =  test [interest_cols]

#print(train.dtypes)
X_train,y_train = train[X_col],train[y_col]
X_test,  y_test = test[X_col] ,test[y_col]


xg_reg = xgb.XGBClassifier(
    objective='binary:logistic',
    # verbosity=2
#     colsample_bytree = 0.3,
#     learning_rate = 0.1,
#     max_depth = 5,
    n_jobs = 6
)

sw = np.where(train['y']==1, (train['y']==0).shape[0]/train.shape[0],(train['y']==1).shape[0]/train.shape[0])
xg_reg.fit(X_train,y_train)


test['y_predict_proba'] = xg_reg.predict_proba(X_test)[:, 1]


ax = sns.distplot(test.loc[test.y==0, "y_predict_proba"], hist=False)
ax = sns.distplot(test.loc[test.y==1, "y_predict_proba"], hist=False)
pyplot.show()




test['predicted_y'] = np.where(test['y_predict_proba']>0.44, 1, 0)
print(metrics.confusion_matrix(test['y'], test['predicted_y']))
print(metrics.log_loss(test['y'], test['y_predict_proba']))

print("#################################################################################")
print("auc",roc_auc_score(test['y'], test['y_predict_proba']))
# average_precision_score(y_test, preds)
fpr, tpr, _ = roc_curve(test['y'], test['y_predict_proba'])
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.show()
print("#################################################################################")

# lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_preds)
# lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
