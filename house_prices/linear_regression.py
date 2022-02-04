from email.mime import nonmultipart
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def preprocess(data):
    cols_to_drop = ['Id']
    for col in data.columns: 
        if data[col].isna().sum() > 100:
            cols_to_drop.append(col)

    data = data.drop(columns=cols_to_drop)
    nonnumeric_cols = []
    for col in data.columns:
        if len(data[col].unique()) < 20 or data[col].dtype.char == 'O':
            data[col] = data[col].factorize()[0]
            nonnumeric_cols.append(col)
    
    data = data.dropna()
    for col in data.columns:
        if col not in nonnumeric_cols and col != 'SalePrice':
            data[col] = (data[col]-data[col].mean())/data[col].std()

    output = data['SalePrice']
    data = data.drop(columns=['SalePrice'])
    return data, output

data = pd.read_csv('data/house-prices-advanced-regression-techniques/train.csv')
data, data_y = preprocess(data)

split = 0.8*data.shape[0]
train_x = data.iloc[:int(split),:].to_numpy()
val_x = data.iloc[int(split):,:].to_numpy()
train_y = data_y.iloc[:int(split)].to_numpy()
val_y = data_y.iloc[int(split):].to_numpy()


b = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x.T,train_x)),train_x.T),train_y)

all_errors = np.zeros(val_x.shape[0])
for i in range(val_x.shape[0]):
    pred_y = np.matmul(train_x[i],b)
    all_errors[i] = abs(pred_y - val_y[i])

m, h = mean_confidence_interval(all_errors)
print(f'{m}+/-{h}')



bars = plt.bar(np.arange(0,len(b)),b)
plt.bar_label(container=bars, labels=np.arange(0,len(b)))
plt.show()
print(b)
