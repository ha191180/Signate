# 申込率を含む顧客リストの作成
customer_list = pd.DataFrame(index=test_X.index, data={"cvr":pred_y3})

# 期待できる収益の計算
customer_list['return'] = 2000 * customer_list['cvr']

# 期待できるROIの計算 期待できる利益/費用*100
customer_list['ROI'] = customer_list['return'] / 300 * 100

# ROIで降順に並べ替え
sorted_customer_list = customer_list.sort_values('ROI', ascending=False)

# ROIが100%以上の顧客idを切り出したアタックリストの作成
attack_list = sorted_customer_list[sorted_customer_list['ROI'] >= 100]

# アタックリストの行数・列数の表示
print( attack_list.shape )

# アタックリストの先頭5行の表示
print( attack_list.head() )