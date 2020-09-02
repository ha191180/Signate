# matplotlib.pyplotのインポート
from matplotlib import pyplot as plt

# roc_curveのインポート
from sklearn.metrics import roc_curve

# 偽陽性率、真陽性率、閾値の計算
# なお、予測結果は以下の変数に代入されているものとします。
# pred_y1：max_depth=2の場合の予測結果
# pred_y2：max_depth=10の場合の予測結果
# pred_y3：max_depth=6の場合の予測結果
# また、それぞれの戻り値を代入する変数は以下とします。
# fpr1,tpr1,thresholds1：max_depth=2の場合の偽陽性率、真陽性率、閾値
# fpr2,tpr2,thresholds2：max_depth=10の場合の偽陽性率、真陽性率、閾値
# fpr3,tpr3,thresholds3：max_depth=6の場合の偽陽性率、真陽性率、閾値
fpr1, tpr1, thresholds1 = roc_curve(test_y, pred_y1) 
fpr2, tpr2, thresholds2 = roc_curve(test_y, pred_y2) 
fpr3, tpr3, thresholds3 = roc_curve(test_y, pred_y3) 

# ラベル名の作成
# なお、それぞれの戻り値を代入する変数は以下とします。
# roc_label1：max_depth=2の場合のラベル名
# roc_label2：max_depth=10の場合のラベル名
# roc_label3：max_depth=6の場合のラベル名
roc_label1='ROC(AUC={:.2}, max_depth=2)'.format(auc1)
roc_label2='ROC(AUC={:.2}, max_depth=10)'.format(auc2)
roc_label3='ROC(AUC={:.2}, max_depth=6)'.format(auc3)

# ROC曲線の作成
plt.plot(fpr1, tpr1, label=roc_label1)
plt.plot(fpr2, tpr2, label=roc_label2)
plt.plot(fpr3, tpr3, label=roc_label3)

# 対角線の作成
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')

# グラフにタイトルを追加
plt.title('ROC')

# グラフのx軸に名前を追加
plt.xlabel('FPR')

# グラフのy軸に名前を追加
plt.ylabel('tpr')

# x軸の表示範囲の指定
plt.xlim(0,1)

# y軸の表示範囲の指定
plt.ylim(0,1)

# 凡例の表示
plt.legend()

# グラフを表示
plt.show()