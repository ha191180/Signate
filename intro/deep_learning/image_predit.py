# ioモジュールのインポート
from skimage import io

# transformモジュールのインポート
from skimage import transform

# 画像の読み込みんで画像の中身を確認する
img = io.imread('img.jpg')
print(img)
print(type(img))

# 画像の大きさを確認
print(img.shape)

# pltのインポート
from matplotlib import pyplot as plt

# 画像の表示
plt.imshow(img)
plt.show()

# 1行3列の複数画像の出力

# 1行3列の1番目
plt.subplot(1,3,1)
plt.imshow(img)

# 1行3列の2番目
plt.subplot(1,3,2)
plt.imshow(img)

# 1行3列の3番目
plt.subplot(1,3,3)
plt.imshow(img)

plt.show()