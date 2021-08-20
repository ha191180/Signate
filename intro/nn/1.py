import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.layers import Dropout
from keras.layers import Sequential
from keras.models import Dense, LSTM


#Unused
def get_t(X, num_date):
    # 入力データをNumPy配列に変換
    X = np.array(X)
    X_t_list = []
    for i in range(len(X) - num_date + 1):
        X_t = X[i:i+num_date, :]
        X_t_list.append(X_t)
    # Numpy配列のreturn
    return np.array(X_t_list)

def get_standardized_t(X, num_date):
    X = np.array(X)
    X_t_list = []
    for i in range(len(X) - num_date + 1):
        X_t = X[i:i+num_date]
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X_t)
        X_t_list.append(X_standardized)
    return np.array(X_t_list)

# データの読み込み
df = pd.read_csv('train.csv')

# Dateのデータ型の変更
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# データの並び替え
df.sort_values(by='Date', ascending=True, inplace=True)

# インデックスの更新
df.set_index(keys='Date', inplace=True)


# 特定のカラムの抽出
df_2columns = df.loc[:, ['Close','Adj Close']]

# インデックスとカラムの入れ替え
df_transposed = df_2columns.transpose()

print(df_transposed)

# 重複カラムの確認
print(df_transposed.duplicated(keep='first'))

# 重複カラムの削除
df.drop(columns=['Adj Close'], inplace=True)

# OpenとCloseの差額を計算
df['Body'] = df['Open'] - df['Close']

# 説明変数をX_dataに格納
X_data = df.drop(columns=['Up'])

# 目的変数をy_dataに格納
y_data = df.loc[:,'Up']

# 学習データおよび検証データと、評価データに80:20の割合で2分割する
X_trainval, X_test, y_trainval, y_test = train_test_split(X_data, y_data, test_size=0.20, shuffle=False)

# 学習データと検証データに75:25の割合で2分割する
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, shuffle=False)

num_date = 5    

# 関数get_standardized_tの呼び出し
X_train_t =  get_standardized_t(X=X_train, num_date=num_date)
X_val_t = get_standardized_t(X=X_val, num_date=num_date)
X_test_t = get_standardized_t(X=X_test, num_date=num_date)

# 目的変数の変形
y_train_t = y_train[4:]
y_val_t = y_val[4:]
y_test_t = y_test[4:]


# ネットワークの各層のサイズの定義
num_l1 = 100
num_l2 = 20
num_output = 1

# Dropoutの割合の定義
dropout_rate = 0.4

# 以下、ネットワークを構築
model = Sequential()
# 第1層
model.add(LSTM(units=num_l1,
                activation='tanh',
                batch_input_shape=(None, X_train_t.shape[1], X_train_t.shape[2])))
model.add(Dropout(dropout_rate))
# 第2層
model.add(Dense(num_l2, activation='relu'))
model.add(Dropout(dropout_rate))
# 出力層
model.add(Dense(num_output, activation='sigmoid'))
# ネットワークのコンパイル
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# モデルの学習の実行（学習の完了までには数秒から数十秒ほど時間がかかります。）
result = model.fit(x=X_train_t , y=y_train_t, epochs=80, batch_size=24, validation_data=(X_val_t, y_val_t))

# 評価データの予測結果の算出
pred_prob = model.predict(X_test_t)

# 予測結果の先頭10件を確認
print('予測結果の先頭10件')
print(pred_prob[:10])

# 評価データの予測結果を0もしくは1に丸め込み
pred = np.round(pred_prob)

# 丸め込んだ予測結果の先頭10件を確認
print('丸め込んだ予測結果の先頭10件')
print(pred[:10])

# 評価データの正解率の計算
accuracy = accuracy_score(y_true=y_test_t, y_pred=pred)

# 評価データの正解率の表示
print('評価データの正解率:', accuracy)
