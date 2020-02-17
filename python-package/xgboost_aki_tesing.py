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

print(xgb.__version__)
#data_dir= '~/projects/AKI/AKI_test_data'
#data_dir= '~/projects/AKI/test'
data_dir='data/'
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
    colsample_bytree = 0.3,
    learning_rate = 0.1,
    max_depth = 5,
    alpha = 10,
    n_estimators = 2,
)

# xg_reg = xgb.XGBRegressor(
#     objective = 'binary:logistic', #'reg:linear',
#     colsample_bytree = 0.3,
#     learning_rate = 0.1,
#     max_depth = 5,
#     alpha = 10,
#     n_estimators = 2,
# )

sw = np.where(train['y']==1, (train['y']==0).shape[0]/train.shape[0],(train['y']==1).shape[0]/train.shape[0])
xg_reg.fit(X_train,y_train, sample_weight=sw)


test['y_predict_proba'] = xg_reg.predict_proba(X_test)[:, 1]


# test.groupby('y')['y_predict_proba'].hist()
# from sklearn import metrics

# loss = metrics.log_loss(y_test, y_predict_proba)
# import seaborn as sns
# sns.distplot(y_predict_proba)

# y_preds = xg_reg.predict(X_test)

# print("#################################################################################")
# print("auc",roc_auc_score(y_test, y_preds))
# # average_precision_score(y_test, preds)
# fpr, tpr, _ = roc_curve(y_test, y_preds)
# pyplot.plot(fpr, tpr, marker='.', label='Logistic')

# print("#################################################################################")

# # lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_preds)
# # lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)


ax = sns.distplot(test.loc[test.y==0, "y_predict_proba"], hist=False)
ax = sns.distplot(test.loc[test.y==1, "y_predict_proba"], hist=False)
pyplot.show()
hist_0, bin_edges_0 = np.histogram(test.loc[test.y==0, "y"].values, normed=True, bins=100)
hist_1, bin_edges_1 = np.histogram(test.loc[test.y==1, "y"].values, normed=True, bins=100)

for i in range(0, 100):
    if hist_1[i] > hist_0[i]:
        cutoff = bin_edges_1[i]
        break


test['predicted_y'] = np.where(test['y_predict_proba']>cutoff, 1, 0)
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
