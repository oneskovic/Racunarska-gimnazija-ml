import torch
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import pickle
import optuna
import plotly

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

def objective(trial):
    train_data, train_y, test_data, test_y = data_from_csv()
    input_size = train_data.shape[1]

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 1, 32)
    layer_cnt = trial.suggest_int('layer_cnt', 1, 5)
    
    layer_sizes = []
    for i in range(layer_cnt):
        layer_sizes.append(trial.suggest_int(f'layer_size_{i}', 1, 2*input_size))

    print(f'Params: {trial.params}')

    layers = []
    layers.append(torch.nn.Linear(input_size, layer_sizes[0]))
    layers.append(torch.nn.ReLU())
    for i in range(1,layer_cnt):
        layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(layer_sizes[-1], 1))

    nn = torch.nn.Sequential(*layers)
    
    criterion = RMSLELoss()
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(100):
        losses = []
        train_data, train_y, test_data, test_y = data_from_csv(split=0.8)
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad()
            y_pred = nn(torch.tensor(train_data.iloc[i:i+batch_size, :].values, dtype=torch.float32))
            y_true = torch.tensor(train_y[i:i+batch_size], dtype=torch.float32).reshape(y_pred.shape)
            loss = criterion(y_pred,y_true)
            losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

        y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32)).flatten().detach().numpy()
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(test_y.values, dtype=torch.float32).flatten()

        loss = criterion(y_pred,y_true)
        print('Epoch {}: Loss {}'.format(epoch, loss.detach().numpy()))
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss

def eval_net(hparams):
    train_data, train_y, test_data, test_y = data_from_csv()
    input_size = train_data.shape[1]
    learning_rate = hparams['learning_rate']
    batch_size = hparams['batch_size']
    layer_cnt = hparams['layer_cnt']

    layer_sizes = []
    for i in range(layer_cnt):
        layer_sizes.append(hparams[f'layer_size_{i}'])

    layers = []
    layers.append(torch.nn.Linear(input_size, layer_sizes[0]))
    layers.append(torch.nn.ReLU())
    for i in range(1,layer_cnt):
        layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(layer_sizes[-1], 1))
    
    nn = torch.nn.Sequential(*layers)

    criterion = RMSLELoss()
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.005, weight_decay=1e-5)
    train_losses = []
    val_losses = []
    for epoch in range(20):
        losses = []
        train_data, train_y, test_data, test_y = data_from_csv(split=0.8)
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad()
            y_pred = nn(torch.tensor(train_data.iloc[i:i+batch_size, :].values, dtype=torch.float32))
            y_true = torch.tensor(train_y[i:i+batch_size], dtype=torch.float32).reshape(y_pred.shape)
            loss = criterion(y_pred,y_true)
            losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

        y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32)).flatten().detach().numpy()
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(test_y.values, dtype=torch.float32).flatten()

        loss = criterion(y_pred,y_true)
        train_loss = np.array(losses).mean()
        print(f'Epoch {epoch} Training Loss {train_loss}')
        print(f'Epoch {epoch} Validation Loss {loss.item()}')
        train_losses.append(train_loss)
        val_losses.append(loss.item())

        error = np.abs(y_pred.detach().numpy() - y_true.detach().numpy())
        #print('R2 score: {}'.format(r2_loss(y_pred, y_true)))
        m,h = mean_confidence_interval(error)
        print('0.95 confidence: {} +- {}'.format(m,h))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    return nn, loss


def main():

    # Uncomment to optimize hparams
    # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(
    #     min_resource=15, max_resource=100, reduction_factor=3))
    # study.optimize(objective, n_trials=200)
    # pickle.dump(study, open('study.pkl', 'wb+'))
    # optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    # plt.show()
    
    # Load hparams
    study = pickle.load(open('study.pkl', 'rb'))
    study.optimize(objective, n_trials=200)
    pickle.dump(study, open('study.pkl', 'wb+'))

    hparams = study.best_params
    print(hparams)
    # nn, loss = eval_net(hparams)
    trial_scores = np.array([study.trials[i].value if study.trials[i].value is not None else np.inf for i in range(len(study.trials))])
    sorted_indices = np.argsort(trial_scores)
    for i in range(10):
        value = study.trials[sorted_indices[i]].value
        params = study.trials[sorted_indices[i]].params
        print('Study {}: {}'.format(i, value))
        print(params)

    # Output predictions for test set
    # test_data, test_y,_,_ = data_from_csv(split=1.0,csv_path='data/house-prices-advanced-regression-techniques/test.csv', should_drop=False)
    # test_data = test_data.sort_index()
    # test_data = torch.tensor(test_data.values, dtype=torch.float32)
    # y_pred = nn(test_data)
    # y_pred = y_pred.detach().numpy()
    # pred_df = pd.DataFrame(y_pred, columns=['SalePrice'])
    # pred_df.index += 1461
    # pred_df.to_csv('pred.csv')

if __name__ == '__main__':
    main()