# 最適なパラメータの表示
print( gcv.best_params_ )

# 最適なパラメータで学習したモデルの取得
best_model = gcv.best_estimator_

# 評価用データの予測
pred_y3 = best_model.predict_proba(test_X)[:,1]

# AUCの計算
auc3 = roc_auc_score(test_y, pred_y3)

# AUCの表示
print ( auc3 )