from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import pandas as pd
import scipy.stats



data = pd.read_csv("data/train.csv", index_col='id')
main_data = pd.read_csv('data/test.csv', index_col='id')
data = pd.get_dummies(data)
#print(data.dtypes)

x = data.drop(columns='y')
x.insert(18,'job_unknown',0)
t = data['y']
#print(x)
train_X, test_X, train_y, test_y = train_test_split(x,t, random_state=0)

model = SVC(kernel='linear', random_state=0, C=10**-5, probability=True, verbose=1)


#SVC_grid = {SVC(): {"C": [10 ** i for i in range(-5, 6)],
#                    "kernel": ["linear", "rbf", "sigmoid"],
#                    "decision_function_shape": ["ovo", "ovr"],
#                    }}
#SVC_grid = {SVC(): {"C": [10 ** i for i in range(-1, 1)],
#                    "kernel": ["linear", "rbf", "sigmoid"],
#                   "decision_function_shape": ["ovo", "ovr"],
#                   }}

#グリッドサーチ
#for model, param in SVC_grid.items():
#    clf = GridSearchCV(model, param, cv=3, verbose=3)
#    clf.fit(train_X, train_y)
#    pred_y = clf.predict(test_X)
#    score = f1_score(test_y, pred_y, average="micro")
#
#    if max_score < score:
#        max_score = score
#        best_param = clf.best_params_
#        best_model = model.__class__.__name__
#
#else:
#    print("サーチ方法:ランダムサーチ")
#print("ベストスコア:{}".format(max_score))
#print("モデル:{}".format(best_model))
#print("パラメーター:{}".format(best_param))

model.fit(train_X, train_y)
pred = model.predict_proba(test_X)[:,1]
auc = roc_auc_score(test_y, pred)
print(auc)

main_data = pd.get_dummies(main_data)
print(train_X.iloc[0,:] )
print(main_data.iloc[0,:])
main_resu = model.predict_proba(main_data)[:,1]
main_resu = pd.DataFrame(main_resu)
main_resu.to_csv('result.csv')
