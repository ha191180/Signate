# ライブラリのインポート
from matplotlib import pyplot as plt
from skimage import io

# 画像読み込み
img = io.imread("cat.jpg")

# 画像切り出し
img_trim = img[100:300,200:400]

# 切り出した画像表示
plt.imshow(img_trim)
plt.show()

# 配列の定義
tmp = np.array([1,2,3,4,5])
# 配列の表示
print(tmp)
# 反転した配列の表示
print(tmp[::-1])

# 画像の読み込み
img = io.imread("cat.jpg")
# 画像の反転
img_inv = img[:,::-1]

# 画像の複数表示
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(img_inv)

plt.show()

# transformのインポート
from skimage import transform

# 画像の読み込み
img = io.imread("cat.jpg")
# 画像のリサイズ
img_resize = transform.resize(img,output_shape=(100,100))
# 画像の表示
plt.imshow(img_resize)
plt.show()

# 画像の読み込み
img = io.imread("cat.jpg")
# 画像の回転
img_rotate = transform.rotate(img,angle=90)
# 画像の表示
plt.imshow(img_rotate)
plt.show()


from skimage import transform

# 画像の読み込み
img = io.imread("cat.jpg")
# 移動位置の定義
rot = transform.AffineTransform(translation=(40,40))

# 画像の移動
img_affine = transform.warp(img,rot)

# 画像の表示
plt.imshow(img_affine)
plt.show()