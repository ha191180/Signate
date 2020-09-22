# ライブラリの読み込み
import torch
import numpy as np

np_y = np.ones([5,3])
dct  = [[1,1,1],[1,1,1],[1,1,1]]

# テンソル化
ts_y = torch.from_numpy(np_y)
ts_y = torch.tensor(dct, dtype = torch.float)

# テンソル出力
print(ts_y)

print(ts_y.size())

print(ts_y[0,0].item())

print(torch.abs(ts_y))
print(torch.mean(ts_y))
print(torch.std(ts_y))

