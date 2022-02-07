import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import pickle

def get_cols_nonnumeric(data, should_print=False):
    nonnumerical_cols = []
    for c in data.columns:
        if data[c].dtype != 'float64' and data[c].dtype != 'int64':
            nonnumerical_cols.append(c)
    if should_print:
        print('{} Non numerical columns: {}'.format(len(nonnumerical_cols), nonnumerical_cols))
    return nonnumerical_cols

def convert_nonnumeric_to_numeric(data):
    nonnumerical_cols = get_cols_nonnumeric(data)
    for c in nonnumerical_cols:
        data[c] = pd.factorize(data[c])[0]

def normalize_data(data, cols_to_skip):
    for c in data.columns:
        if c not in cols_to_skip:
            data[c] = (data[c] - data[c].mean()) / data[c].std()
    return data


def fill_na(data):
    for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
            "FireplaceQu","GarageQual","GarageCond","PoolQC"]:
        if col not in data.columns:
            continue
        data[col]= data[col].map({"Gd": 4 , "TA": 3, "Ex": 5, "Fa":2, "Po":1})

    for col in ('Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
        if col not in data.columns:
            continue
        data[col]=data[col].fillna('Jaje')
    for col in ('Electrical','MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
        if col not in data.columns:
            continue
        data[col]=data[col].fillna(data[col].mode()[0])
    
    for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
            'GarageYrBlt','GarageCars','GarageArea'):
        if col not in data.columns:
            continue
        data[col]=data[col].fillna(0)

    data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].mean())
    
def data_from_csv(should_print = False, should_drop = True, split = 0.9, csv_path = 'data/house-prices-advanced-regression-techniques/train.csv'):
    data = pd.read_csv(csv_path)
    cols_to_drop = ['Id',]
    data.drop(cols_to_drop, axis=1, inplace=True)
    
    if should_print:
        for c in data.columns:
            print('Percent nan in column {}: {}'.format(c, data[c].isnull().sum() / len(data)))

    if should_print:
        print('Replaced nan --------------------')
    fill_na(data)
    if should_print:
        for c in data.columns:
            print('Percent nan in column {}: {}'.format(c, data[c].isnull().sum() / len(data)))

    row_start = data.shape[0]
    if should_drop:
        data = data.dropna()
    rows_dropped = row_start - data.shape[0]
    if should_print:
        print('Dropped {} rows with nan values which is {}%'.format(rows_dropped, rows_dropped / row_start))

    if should_drop:
        data = data[data['GrLivArea']<4000]

    data.reset_index(drop=True, inplace=True)

    nonnumeric_cols = get_cols_nonnumeric(data, should_print=should_print)
    convert_nonnumeric_to_numeric(data)
    get_cols_nonnumeric(data, should_print)

    data = data.astype('float32')
    data = normalize_data(data, nonnumeric_cols + ['SalePrice'])

    #price_col = data.pop('SalePrice')

    # data_np = data.values
    # cov = np.dot(data_np.T, data_np) / data_np.shape[0]
    # U,S,V = np.linalg.svd(cov)
    # Xrot = np.dot(data_np, U)
    # Xwhite = Xrot / np.sqrt(S + 1e-5)
    
    # cov = np.dot(Xwhite.T, Xwhite) / Xwhite.shape[0]
    # U,S,V = np.linalg.svd(cov)
    # Xrot = np.dot(Xwhite, U)
    # Xrot_reduced = np.dot(Xwhite, U[:,:30])

    # data = pd.DataFrame(Xrot_reduced)
    if should_drop:
        data = data.dropna()

    #data['SalePrice'] = price_col

    # Export to csv
    data.to_csv('data_normalized.csv', index=False)

    # train_data = data.sample(frac=split, random_state=0)

    # if 'SalePrice' in train_data.columns:
    #     train_y = train_data['SalePrice'].to_numpy()
    #     train_data.drop('SalePrice', axis=1, inplace=True)
    # else:
    #     train_y = None
    
    # test_data = data.drop(train_data.index)

    # if 'SalePrice' in test_data.columns:
    #     test_y = test_data['SalePrice']
    #     test_data.drop('SalePrice', axis=1, inplace=True)
    # else:
    #     test_y = None

    # return train_data, train_y, test_data, test_y


data_from_csv()