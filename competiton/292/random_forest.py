import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("data/train.csv", index_col='id')
main_data = pd.read_csv('data/test.csv', index_col='id')
data = pd.get_dummies(data)

x = data.drop(columns='y')
x.insert(18,'job_unknown',0)
t = data['y']
train_X, test_X, train_y, test_y = train_test_split(x,t, random_state=0)

model = RandomForestRegressor()

params = {
          'n_estimators': list(range(20, 101, 10)),
          'max_depth':[7,8,10,20,30],
          }

gcv = GridSearchCV(model, params, cv=5,scoring = 'roc_auc',verbose=3)

gcv.fit(train_X, train_y)

print(gcv.best_params_)


model = gcv.best_estimator_

main_data = pd.get_dummies(main_data)
main_resu = model.predict_proba(main_data)[:,1]
main_resu = pd.DataFrame(main_resu)
main_resu.to_csv('result.csv')

