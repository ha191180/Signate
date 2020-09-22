# ライブラリのインポート
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

# モデルの定義
model = Sequential()
# 1. 畳み込み層(フィルタサイズ=6, カーネルサイズ=(3,3), 入力サイズ=(96,96,3))  
model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=(96, 96, 3)))
# 2. 活性化関数(Sigmoid)  
model.add(Activation("sigmoid"))
# 3. プーリング層(プーリングサイズ=(2,2))  
model.add(MaxPooling2D(pool_size=(2, 2)))
# 4. 畳み込み層(フィルタサイズ=12, カーネルサイズ=(3,3))  
model.add(Conv2D(filters=12, kernel_size=(3, 3)))
# 5. 活性化関数(Sigmoid)  
model.add(Activation("sigmoid"))
# 6. プーリング層(プーリングサイズ=(2,2))  
model.add(MaxPooling2D(pool_size=(2, 2)))
# 7. 平坦化()  
model.add(Flatten())
# 8. 全結合層(出力=120)  
model.add(Dense(units=120))
# 9. 活性化関数(Sigmoid)  
model.add(Activation("sigmoid"))
# 10. 全結合層(出力=60)  
model.add(Dense(units=60))
# 11. 活性化関数(Sigmoid)  
model.add(Activation("sigmoid"))
# 12. 全結合層(出力=4)  
model.add(Dense(units=4))
# 13. 活性化関数(Softmax)
model.add(Activation("softmax"))

# モデルの確認
model.summary()

from keras import optimizers

model.compile(loss = 'categorical_crossentropy', optimizer=optimizers.SGD(lr=  0.5), metrics=["accuracy"])

history_log = model.fit(X_train, y_train, batch_size=10, epochs=5, validation_data=(X_val, y_val))
