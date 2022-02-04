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
    
    test_data = data.drop(train_data.index)

    if 'SalePrice' in test_data.columns:
        test_y = test_data['SalePrice']
        test_data.drop('SalePrice', axis=1, inplace=True)
    else:
        test_y = None

    return train_data, train_y, test_data, test_y

def dummy_func(x,y):
    return np.power((x*x + y -11),2) + np.power(x+y*y-7,2)

def dummy_func2(x):
    d = 0.5
    m = 0.0
    return 1.0/(d*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-m)/d)**2)

def data_from_random_gen():
    a = -5 
    b = 5
    n = 1000
    t = 100
    dimen = 1
    train_data = (b - a) * np.random.ranf((n,dimen)) + a
    train_y = [dummy_func2(x) for x in train_data]
    test_data = (b - a) * np.random.ranf((t,dimen)) + a
    test_y = [dummy_func2(x) for x in test_data]
    
    return pd.DataFrame(train_data, columns=['x']), pd.DataFrame(train_y, columns=['out']), pd.DataFrame(test_data, columns=['x']), pd.DataFrame(test_y, columns=['out'])

def train():
    train_data, train_y, test_data, test_y = data_from_csv(True)
    batch_size = np.random.randint(3,12)
    input_size = train_data.shape[1]
    nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, max(10,input_size//2)),
        torch.nn.ReLU(),
        torch.nn.Linear(max(10,input_size//2), max(10,input_size//4)),
        torch.nn.ReLU(),
        torch.nn.Linear(max(10,input_size//4), 1)
    )
    
    criterions = [torch.nn.L1Loss(reduction='mean'), RMSLELoss(), torch.nn.MSELoss()]
    criterion = np.random.choice(criterions)
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.005, weight_decay=1e-5)
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
        #print('R2 score: {}'.format(r2_loss(y_pred, y_true)))
        m,h = mean_confidence_interval(error)
        print('0.95 confidence: {} +- {}'.format(m,h))

    train_data_df, train_y, test_data, test_y = data_from_csv(split=1.0)
    train_data = torch.tensor(train_data_df.values, dtype=torch.float32)
    train_data.requires_grad = True
    y_pred = nn(train_data)

    for i in range(0, train_data.shape[0]):
        y_pred[i].backward(retain_graph=True)
    
    avg_grads = torch.mean(torch.abs(train_data.grad), dim=0).detach().numpy()
    # bars = plt.bar(range(avg_grads.shape[0]), avg_grads)
    # plt.bar_label(container=bars, labels=np.arange(0,len(avg_grads)))
    # plt.show()

    cols_where_grad_small = [ train_data_df.columns[i] for i,x in enumerate(avg_grads) if x < 500]
    return cols_where_grad_small, nn, loss

def main():
    # small_grad_cols_cnt = dict()
    # for i in range(1):
    #     cols, _ = train()
    #     for col in cols:
    #         small_grad_cols_cnt[col] = small_grad_cols_cnt.get(col, 0) + 1
    # print(small_grad_cols_cnt)

    nets = []
    losses = []
    for i in range(100):
        print('Iteration {}'.format(i))
        _, nn, loss = train()
        nets.append(nn)
        losses.append(loss)
    
    test_data, test_y,_,_ = data_from_csv(split=1.0,csv_path='data/house-prices-advanced-regression-techniques/train.csv', should_drop=False)
    test_data = test_data.sort_index()
    test_data = torch.tensor(test_data.values, dtype=torch.float32)

    best_net = nets[np.argmin(losses)]
    is_net_chosen = np.zeros((len(nets),1))
    is_net_chosen[np.argmin(losses)] = 1

    preds = np.array([nn(test_data).detach().flatten().numpy() for nn in nets])
    for _ in range(9):
        corr = np.corrcoef(preds)
        scores = np.matmul(corr,is_net_chosen)
        for i in range(len(is_net_chosen)):
            if is_net_chosen[i]:
                scores[i] = np.inf
        
        new_net = np.argmin(scores)
        is_net_chosen[new_net] = 1

    net_index = 0
    for i in range(len(preds)):
        if is_net_chosen[i]:
            pickle.dump(nets[i], open('trained_nets/net_{}.pkl'.format(net_index), 'wb+'))
            net_index += 1
            plt.scatter(range(len(preds[i])),preds[i])
    
    plt.show()


    # y_pred = nn(test_data)
    # y_pred = y_pred.detach().numpy()
    # pred_df = pd.DataFrame(y_pred, columns=['SalePrice'])
    # pred_df.to_csv('pred.csv')

if __name__ == '__main__':
    main()