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

class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def data_from_csv(split, csv_path='data_normalized2.csv'):
    data = pd.read_csv(csv_path)

    train_data = data.sample(frac=split, random_state=0)

    if 'SalePrice' in train_data.columns:
        train_y = train_data['SalePrice']
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
    train_data, train_y, test_data, test_y = data_from_csv(0.8)
    batch_size = 4
    learning_rate = 0.005
    input_size = train_data.shape[1]
    nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    
    criterion = RMSLELoss()
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(100):
        losses = []
        train_data, train_y, test_data, test_y = data_from_csv(split=0.8)
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad()
            y_pred = nn(torch.tensor(train_data.iloc[i:i+batch_size, :].values, dtype=torch.float32))
            y_true = torch.tensor(train_y.iloc[i:i+batch_size].values, dtype=torch.float32).reshape(y_pred.shape)
            loss = criterion(y_pred,y_true)
            losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

        y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32)).flatten().detach().numpy()
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
    bars = plt.bar(range(avg_grads.shape[0]), avg_grads)
    plt.bar_label(container=bars, labels=np.arange(0,len(avg_grads)))
    plt.show()

    cols_where_grad_small = [ train_data_df.columns[i] for i,x in enumerate(avg_grads) if x < 500]
    return cols_where_grad_small, nn, loss

def main():
    # small_grad_cols_cnt = dict()
    # for i in range(1):
    #     cols, _ = train()
    #     for col in cols:
    #         small_grad_cols_cnt[col] = small_grad_cols_cnt.get(col, 0) + 1
    # print(small_grad_cols_cnt)   
    _, nn, loss = train()
    
    # y_pred = nn(test_data)
    # y_pred = y_pred.detach().numpy()
    # pred_df = pd.DataFrame(y_pred, columns=['SalePrice'])
    # pred_df.to_csv('pred.csv')

if __name__ == '__main__':
    main()