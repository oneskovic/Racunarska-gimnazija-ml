import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import pickle

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

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

def r2_loss(output, target):
    with torch.no_grad():
        output = output.flatten()
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

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
    cols_to_drop = ['Id','Street', 'Utilities' , 'LotArea','LandContour','LotConfig','Condition2','Exterior2nd','BsmtHalfBath','HalfBath','Functional','3SsnPorch','PoolArea','PoolQC','MiscVal','SaleType','SaleCondition']
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

    nonnumeric_cols = get_cols_nonnumeric(data, should_print=should_print)
    convert_nonnumeric_to_numeric(data)
    get_cols_nonnumeric(data, should_print)

    data = data.astype('float32')
    data = normalize_data(data, nonnumeric_cols + ['SalePrice'])
    if should_drop:
        data = data.dropna()

    train_data = data.sample(frac=split, random_state=0)

    if 'SalePrice' in train_data.columns:
        train_y = train_data['SalePrice'].to_numpy()
        train_data.drop('SalePrice', axis=1, inplace=True)
    else:
        train_y = None

    ensamble_net_cnt = 10
    preds = []
    for i in range(ensamble_net_cnt):
        nn = pickle.load(open('trained_nets/net_{}.pkl'.format(i), 'rb'))
        preds.append(nn(torch.tensor(train_data.values, dtype=torch.float32)))

    data_cols = list(train_data.columns)
    for i in range(ensamble_net_cnt):
        train_data['pred_{}'.format(i)] = preds[i].detach().numpy()
    
    # train_data.drop(data_cols, axis=1, inplace=True)

    train_data = normalize_data(train_data, nonnumeric_cols)

    test_data = data.drop(train_data.index)

    if 'SalePrice' in test_data.columns:
        test_y = test_data['SalePrice']
        test_data.drop('SalePrice', axis=1, inplace=True)
    else:
        test_y = None

    preds = []
    for i in range(ensamble_net_cnt):
        nn = pickle.load(open('trained_nets/net_{}.pkl'.format(i), 'rb'))
        preds.append(nn(torch.tensor(test_data.values, dtype=torch.float32)))
    
    data_cols = list(test_data.columns)
    for i in range(ensamble_net_cnt):
        test_data['pred_{}'.format(i)] = preds[i].detach().numpy()

    # test_data.drop(data_cols, axis=1, inplace=True)

    test_data = normalize_data(test_data, nonnumeric_cols)

    return train_data, train_y, test_data, test_y

def train():
    train_data, train_y, test_data, test_y = data_from_csv(True)
    batch_size = 4
    input_size = train_data.shape[1]
    nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, max(10,input_size//2)),
        torch.nn.ReLU(),
        torch.nn.Linear(max(10,input_size//2), max(10,input_size//4)),
        torch.nn.ReLU(),
        torch.nn.Linear(max(10,input_size//4), 1)
    )
    

    criterion = RMSLELoss()
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.01, weight_decay=1e-5)
    for epoch in range(100):
        losses = []
        train_data, train_y, test_data, test_y = data_from_csv(split=0.8)
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad()
            y_pred = nn(torch.tensor(train_data.iloc[i:i+batch_size, :].values, dtype=torch.float32))
            y_true = torch.tensor(train_y[i:i+batch_size], dtype=torch.float32)
            loss = criterion(y_pred,y_true)
            losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

        y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32)).flatten().detach().numpy()
        # y_pred = torch.tensor(np.expm1(y_pred))
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(test_y.values, dtype=torch.float32).flatten()

        loss = criterion(y_pred,y_true)
        print('Epoch {}: loss {}'.format(epoch, loss.item()))
        error = np.abs(y_pred.detach().numpy() - y_true.detach().numpy())
        m,h = mean_confidence_interval(error)
        print('0.95 confidence: {} +- {}'.format(m,h))

    train_data_df, train_y, test_data, test_y = data_from_csv(split=1.0)
    train_data = torch.tensor(train_data_df.values, dtype=torch.float32)
    train_data.requires_grad = True
    y_pred = nn(train_data)

    for i in range(0, train_data.shape[0]):
        y_pred[i].backward(retain_graph=True)
    
    avg_grads = torch.mean(torch.abs(train_data.grad), dim=0).detach().numpy()

    cols_where_grad_small = [ train_data_df.columns[i] for i,x in enumerate(avg_grads) if x < 500]
    return cols_where_grad_small, nn, loss

def main():
    cols_where_grad_small, nn, loss = train()
    print(cols_where_grad_small)
    print(loss)
    print(nn)

if __name__ == '__main__':
    main()