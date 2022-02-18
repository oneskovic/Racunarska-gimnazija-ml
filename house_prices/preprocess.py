import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
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
        one_hot = pd.get_dummies(data[c], prefix=c)
        data.drop(c, axis=1, inplace=True)
        data = data.join(one_hot)
        # data[c] = pd.factorize(data[c])[0]+1
    return data

def normalize_data(data, cols_to_skip):
    for c in data.columns:
        if c not in cols_to_skip:
            data[c] = (data[c] - data[c].mean()) / (data[c].std()+1e-7)
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
        data[col]=data[col].fillna('-')
    for col in ('Electrical','MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
        if col not in data.columns:
            continue
        data[col]=data[col].fillna(data[col].mode()[0])
    
    for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
            'GarageYrBlt','GarageCars','GarageArea'):
        if col not in data.columns:
            continue
        data[col]=data[col].fillna(0)

    data['Utilities']=data['Utilities'].fillna('AllPub')
    data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].mean())
    
 
def categorical_col(data, col):
    if (len(data[col].unique()) < 20 or data[col].dtype.char == 'O'):
        return True
    return False
 
def numerical_col(data, col):
    if (len(data[col].unique()) >= 20 and data[col].dtype in [np.int64, np.float64]):
        return True
    return False

def preprocess(data):
    cols_to_drop = ['Id']
    
    data = data.drop(columns=cols_to_drop)
 
    year_cols = []
    for col in data.columns:
        if 'Yr' in col or 'Year' in col:
            year_cols.append(col)
    
    for col in year_cols:
        data[col] = data['YrSold'] - data[col]
        data[col] = data[col].fillna(data[col].median())
 
    data = data.drop(columns = ['YrSold'])
 
 
    categorical_cols = []
    data = data.reset_index(drop=True)
    for col in data.columns:
        if categorical_col(data, col) and not col in year_cols:
            categorical_cols.append(col)
            data[col] = data[col].fillna('Missing')
 
    quantitative_cols = []  
    data = data.reset_index(drop=True)
    for col in data.columns:
        if numerical_col(data, col) and col != 'SalePrice' and col not in year_cols:
            quantitative_cols.append(col)
            data[col] = data[col].fillna(0)
            data[col] = (data[col]-data[col].mean())/data[col].std() # vladin komentar -> Z normalisation
 
    print(quantitative_cols)
 
    print('Columns that don\'t belong to neihter of the made categories:')
    for col in data.columns:
        if col not in quantitative_cols and col not in categorical_cols and col not in year_cols:
            print('\t', col)
 
 
    #data = data.dropna()
 
    enc = OneHotEncoder(drop='if_binary')
    enc_data = pd.DataFrame(enc.fit_transform(data[categorical_cols].astype(str)).toarray())
 
    data = data.join(enc_data)
    data = data.drop(columns=categorical_cols)
 
    print('quantitative_cols =', len(quantitative_cols))
    print('categorical_cols =', len(categorical_cols))
    print('year_cols = ', len(year_cols))
 
    return data

def data_from_csv(should_print = False, should_drop = True, split = 0.9, csv_path = 'data/house-prices-advanced-regression-techniques/train.csv'):
    #data1 = pd.read_csv('data/house-prices-advanced-regression-techniques/AmesHousing.csv')
    data2 = pd.read_csv('data/house-prices-advanced-regression-techniques/train.csv')
    test_data = pd.read_csv('data/house-prices-advanced-regression-techniques/test.csv')
    #data1.columns = data1.columns.str.replace(' ', '')
    #data1.columns = data1.columns.str.replace('/', '')
    #data1.drop(['PID', 'Order'], axis=1, inplace=True)
    sale_price = data2.pop('SalePrice')
    data2.drop(['Id'], axis=1, inplace=True)
    test_data.drop(['Id'], axis=1, inplace=True)

    # data = pd.concat([data1, data2], axis=0)
    data = pd.concat([data2, test_data], axis=0)
    # data = pd.read_csv(csv_path)
    # cols_to_drop = ['Id',]
    # data.drop(cols_to_drop, axis=1, inplace=True)
    
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

    data.reset_index(drop=True, inplace=True)

    nonnumeric_cols = get_cols_nonnumeric(data, should_print=should_print)
    data = convert_nonnumeric_to_numeric(data)
    get_cols_nonnumeric(data, should_print)

    # Remove test data from data
    data = data.iloc[:data2.shape[0]]
    data['SalePrice'] = sale_price

    if should_drop:
        data = data[data['GrLivArea']<4000]

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
    data.to_csv('data_normalized3.csv', index=False)

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


# data_from_csv()
df_train = pd.read_csv('data/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('data/house-prices-advanced-regression-techniques/test.csv')
y = df_train.pop('SalePrice')
 
df_concat = pd.concat([df_train, df_test])
 
df_processed = preprocess(df_concat)
df_train = df_processed.iloc[0:df_train.shape[0]]
df_test =  df_processed.iloc[df_train.shape[0]:]

df_train['SalePrice'] = y
df_train.to_csv('train_processed.csv', index=False)
df_test.to_csv('test_processed.csv', index=False)