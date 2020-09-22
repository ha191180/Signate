# Sequentialのインポート
from keras.models import Sequential 

# Dense , Activationのインポート
from keras.layers import Dense
from keras.layers import Activation


# ライブラリのインポート
from keras.models import Sequential
from keras.layers import Dense

# モデルの定義
model = Sequential()
model.add(Dense(10, input_shape = (51, )))
model.add(Dense(10, input_shape = (10, )))
model.add(Dense(1 , input_shape = (10, )))


# ライブラリのインポート
from keras.models import Sequential
from keras.layers import Dense, Activation

# 活性化関数の追加
model = Sequential()
model.add(Dense(10, input_shape=(51,)))
model.add(Activation("tanh"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))


# ライブラリのインポート
from keras.models import Sequential
from keras.layers import Dense, Activation

# 活性化関数の追加
model = Sequential()
# 中間層①
model.add(Dense(150, input_shape=(101,)))
model.add(Activation("sigmoid"))
# 中間層②
model.add(Dense(200))
model.add(Activation("tanh"))
# 中間層③
model.add(Dense(100))
model.add(Activation("relu"))
# 出力層④
model.add(Dense(10))
model.add(Activation("softmax"))

# モデルの確認
model.summary()