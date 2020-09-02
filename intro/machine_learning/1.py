# pandasのインポート
import pandas as pd

# データの読み込み
df = pd.read_csv('data.csv', index_col='id')

# データのダミー変数化
df = pd.get_dummies(df)

# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X, data_y = df.drop('y', axis=1), df['y']

# train_test_splitのインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25, random_state=0)

# 評価関数のインポート
from sklearn.metrics import roc_auc_score

# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# 決定木モデルの準備
tree = DT(max_depth=10, random_state=0)

# 決定木モデルの学習
tree.fit(train_X, train_t)

# 評価用データの予測
pred_y2 = tree.predict(test_X)[:,1]

# AUCの計算
auc2 = roc_auc_score(test_y, pred_y2)

# 評価結果の表示
print( auc2  )