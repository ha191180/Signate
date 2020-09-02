from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import roc_auc_score
import pandas as pd



data = pd.read_csv("data/train.csv", index_col='id')
main_data = pd.read_csv('data/test.csv', index_col='id')
data = pd.get_dummies(data)

x = data.drop(columns='y')
x.insert(18,'job_unknown',0)
t = data['y']
print(x)
train_x, test_x, train_t, test_t = train_test_split(x,t, random_state=0)

tree = DT(random_state=0)
parameters = {'max_depth':[8],
              'min_samples_leaf':[14],
              'min_samples_split':[2],
              }

gcv = GridSearchCV(tree, parameters, cv = 5, scoring = 'roc_auc')

gcv.fit(train_x, train_t)


print(gcv.best_params_)

model = gcv.best_estimator_
pred = model.predict_proba(test_x)[:,1]
auc = roc_auc_score(test_t, pred)
print(auc)

main_data = pd.get_dummies(main_data)
print(train_x.iloc[0,:] )
print(main_data.iloc[0,:])
main_resu = model.predict_proba(main_data)[:,1]
main_resu = pd.DataFrame(main_resu)
main_resu.to_csv('result.csv')