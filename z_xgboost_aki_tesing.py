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
data_dir='/home/lpatel/projects/AKI/data'
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
    colsample_bytree = 0.05,
    learning_rate = 0.05,
    max_depth = 9,
    n_jobs = 6
)

sw = np.where(train['y']==1, (train['y']==0).shape[0]/train.shape[0],(train['y']==1).shape[0]/train.shape[0])
xg_reg.fit(X_train,y_train,sample_weight=sw)


test['y_predict_proba'] = xg_reg.predict_proba(X_test)[:, 1]


ax = sns.distplot(test.loc[test.y==0, "y_predict_proba"], hist=False)
ax = sns.distplot(test.loc[test.y==1, "y_predict_proba"], hist=False)
pyplot.show()

# hist_0, bin_0 = np.histogram(test.loc[test.y==0, "y_predict_proba"].values, normed=True, bins=10)
# hist_1, bin_1 = np.histogram(test.loc[test.y==1, "y_predict_proba"].values, normed=True, bins=10)

# for i in range(10):
#     if hist_1[i]> hist_0[i]:
#         cutoff = bin_1[i]

# print("cutoff: ", cutoff)
# test['predicted_y'] = np.where(test['y_predict_proba']>= cutoff, 1, 0)
# print(metrics.confusion_matrix(test['y'], test['predicted_y']))
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

"""
p vector used: 
0.221717 0.127305 0.0555893 0.158098 0.0726475 0.0177584 0.00131976 0.13385 0.0149683 0.00473002 0.0398985 0.0119609 0.0469865 0.00297053 0.000100141 0.0167074 0.0178693 0.0275493 0.0275509 0.000422355 
3 7 4 1 0 2 
[[254691     26]
 [   900     35]]
0.021429541099158615
#################################################################################
auc 0.7422536144181319
"""

"""
p vector used: 
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
6 17 14 12 19 8 
[[254703     14]
 [   915     20]]
0.02128335785405845
#################################################################################
auc 0.7678128871931036
#################################################################################
"""

"""
p vector used: 
0.013 0.027 0.027 0.027 0.04 0.013 0.013 0.04 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 
11 19 13 15 2 17 
[[254708      9]
 [   921     14]]
0.021392834924755747
#################################################################################
auc 0.7618471723646578
#################################################################################
"""

"""
sw
p vector used: 
0.013 0.027 0.027 0.027 0.04 0.013 0.013 0.04 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 
2 11 14 19 10 9 
[[254712      5]
 [   924     11]]
0.02146126935319618
#################################################################################
auc 0.7624009231257783
#################################################################################
"""

"""
xg_reg = xgb.XGBClassifier(
    objective='binary:logistic',
    # verbosity=2
    colsample_bytree = 0.9,
    learning_rate = 0.05,
    max_depth = 9,
    n_jobs = 6
)

p vector used: 
0.013 0.027 0.027 0.027 0.04 0.013 0.013 0.04 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 
7 14 15 16 4 19 10 8 9 17 18 11 3 1 13 2 12 6 
0.022600546061386447
#################################################################################
auc 0.7674201896583184
#################################################################################


p vector used: 
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
19 10 12 4 15 3 18 16 14 1 11 17 9 13 0 7 2 5 
0.02255969416546546
#################################################################################
auc 0.7676620917596312
#################################################################################

p vector used: 
0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 
16 4 0 11 15 18 2 13 12 6 1 3 9 14 10 7 5 17 
0.022549960669464145
#################################################################################
auc 0.7657709754806208
"""
"""
colsample_bytree is higher, so it try to find best split using most features(0.9)(takes more time), 
so random sampling and weighted sampling become the almost the same and results are alsmost the same
"""


"""
xg_reg = xgb.XGBClassifier(
    objective='binary:logistic',
    # verbosity=2
    colsample_bytree = 0.1,
    learning_rate = 0.05,
    max_depth = 9,
    n_jobs = 6

p vector used: 
0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 
8 3 
0.02366628350256921
#################################################################################
auc 0.7435020041850366
#################################################################################

p vector used: 
0.013 0.027 0.027 0.027 0.04 0.013 0.013 0.04 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 0.067 
6 15 
0.024202204975577565
#################################################################################
auc 0.7545921583645341
#################################################################################

# p vector used: 
# 0.221717 0.127305 0.0555893 0.158098 0.0726475 0.0177584 0.00131976 0.13385 0.0149683 0.00473002 0.0398985 0.0119609 0.0469865 0.00297053 0.000100141 0.0167074 0.0178693 0.0275493 0.0275509 0.000422355 
# 10 1 
# 0.023189493944076284
# #################################################################################
# auc 0.7447824458806428
# #################################################################################`
"""

'''
so we want split with low number of variable (hence speed will be faster), wighted sample will produce better results compare to random.
'''

