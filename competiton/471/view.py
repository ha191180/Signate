import numpy as np
import pandas as pd
import math


DATA_DIR = './data/'


train = pd.read_csv(DATA_DIR + "train.csv")
test = pd.read_csv(DATA_DIR + "test.csv")

train_x = train.drop('judgement', axis=1)

print(train_x.isnull().sum())

d = {}


for i in range(len(train_x)):
	if type(train_x.iloc[i,2]) is str:
		for c in (train_x.iloc[i,2]):
			if (c in d.keys()):
				d[c] = d[c] + 1
			else:
				d[c] = 1
	for c in (train_x.iloc[i,1]):
		if (c in d.keys()):
			d[c] = d[c] + 1
		else:
			d[c] = 1


res = pd.DataFrame(list(d.items()),columns=['Fig', 'Num'])
res.to_csv('out/garbled.csv')

							

