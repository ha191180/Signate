import numpy as np
from optuna.integration import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pd.read_csv("data/train.csv", index_col='id')
main_data = pd.read_csv('data/test.csv', index_col='id')
data = pd.get_dummies(data)

x = data.drop(columns='y')
x.insert(18,'job_unknown',0)
t = data['y']
train_X, test_X, train_y, test_y = train_test_split(x,t, random_state=0)


lgb_train = lgb.Dataset(train_X, train_y)
lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)

params = {
  'task' : 'train',
  'boosting_type' : 'gbdt',
  'objective' : 'binary',
  'metric' : 'binary_logloss',
#  'num_class' : 3,
#  'learning_rate' : 0.1,
#  'num_leaves' : 23,
#  'min_data_in_leaf': 1,
#  'num_iteration': 100,
#  'verbose': 1
}

gbm = lgb.train(
  params,
  lgb_train,
#  num_boost_round=50,
  valid_sets=lgb_eval,
#  early_stopping_rounds=10
  )
y_pred = gbm.predict(test_X, num_iteration=gbm.best_iteration)
print("guess:")
for i in y_pred:
  print("\t", i )
#y_pred = np.argmax(y_pred, axis=1)
#for i in y_pred:
#  print("\t", i)

print("score:", roc_auc_score(test_y, y_pred))


main_data = pd.get_dummies(main_data)
main_resu = gbm.predict(main_data, num_iteration=gbm.best_iteration)
main_resu = pd.DataFrame(main_resu)
main_resu.to_csv('result_LGBM.csv')

