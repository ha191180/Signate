# ライブラリのインポート
import numpy as np
from skimage import io
import glob

# ファイル名取り出し
files = glob.glob("train_*.jpg")

#画像読み込み
train_imgs = []
for fname in files:

    # 画像読み込み
    img = io.imread(fname)
    
    # train_imgsに今読み込んだ画像を格納
    train_imgs.append(img)

# train_imgsの型を出力
print(type(train_imgs))

# train_imgsをNumPy配列化し、その型と形状を出力
train_imgs = np.array(train_imgs)
print(type(train_imgs) , train_imgs.shape)

# ライブラリのインポート
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
import glob

# ファイル名の取り出し
files = (glob.glob("train_*.jpg"))

# 画像の読み込み
train_imgs = []
for fname in files:
    img = io.imread(fname)
    train_imgs.append(img)
train_imgs = np.array(train_imgs)

# 画像表示
plt.figure(figsize=(12,5))
for i in range(10):

    # 2行5列のi+1番目の画像を設定
    plt.subplot(2,5,i+1)
    
    # 画像の表示
    plt.imshow(train_imgs[i])
   
plt.show()

# train_imgsを正規化
train_imgs = train_imgs/255

# train_imgsを出力
print(train_imgs)

# ライブラリのインポート
import pandas as pd
import glob

# 画像のファイル名を取得
files = (glob.glob("train_*.jpg"))

labels=[]
for fname in files:
    a,b,c = fname.split('_')
    labels.append(b)

# one-hot-encoding
labels_dummy = pd.get_dummies(labels)
print(labels_dummy)


import numpy as np

# (1)
X_tr, X_val = np.split(train_imgs,[8])
# (2)
y_tr, y_val = np.split(labels_dummy,[8])

print(X_tr.shape, X_val.shape)
print(y_tr.shape, y_val.shape)