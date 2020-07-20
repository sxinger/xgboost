# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # print feature importance for each iteration

# +
import xgboost as xgb

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def MyCallback():
    def callback(env):
        print(env.model.get_score(importance_type='weight'))
    return callback

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {'objective':'reg:squarederror', 'eval_metric': 'rmse'}

bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dtest, 'test')],
        callbacks=[MyCallback()])
# -


