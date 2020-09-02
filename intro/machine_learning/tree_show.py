# 最適なパラメータで学習したモデルの取得
best_model = gcv.best_estimator_

# 決定木描画ライブラリのインポート
from sklearn.tree import export_graphviz

# 決定木グラフの出力
export_graphviz(best_model, out_file="tree.dot", feature_names=train_X.columns, class_names=["0","1"], filled=True, rounded=True)

# 決定木グラフの表示
from matplotlib import pyplot as plt
from PIL import Image
import pydotplus
import io
g = pydotplus.graph_from_dot_file(path="tree.dot")
gg = g.create_png()
img = io.BytesIO(gg)
img2 = Image.open(img)
plt.figure(figsize=(img2.width/100, img2.height/100), dpi=100)
plt.imshow(img2)
plt.axis("off")
plt.show()