import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import pickle
from pytorch_forecasting.metrics import MAPE


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

def data_from_csv(split, csv_path='train_processed.csv'):
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
    batch_size = 128
    learning_rate = 0.01
    input_size = train_data.shape[1]
    nn = torch.nn.Sequential(
        torch.nn.Linear(input_size, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1)
    )
    
    criterion = RMSLELoss()
    mape = MAPE(reduction='mean')
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    train_data, train_y, test_data, test_y = data_from_csv(split=0.8)

    train_losses = []
    test_losses = []

    for epoch in range(150):
        # Shuffle training and test set
        perm = np.random.permutation(train_data.shape[0])
        train_data = train_data.iloc[perm]
        train_y = train_y.iloc[perm]
        perm2 = np.random.permutation(test_data.shape[0])
        test_data = test_data.iloc[perm2]
        test_y = test_y.iloc[perm2]
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad()
            y_pred = nn(torch.tensor(train_data.iloc[i:i+batch_size, :].values, dtype=torch.float32))
            y_true = torch.tensor(train_y.iloc[i:i+batch_size].values, dtype=torch.float32).reshape(y_pred.shape)
            loss = criterion(y_pred,y_true)
            train_losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()

            with torch.no_grad():

                y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32)).flatten().detach().numpy()
                y_pred = torch.tensor(y_pred)
                y_true = torch.tensor(test_y.values, dtype=torch.float32).flatten()
                #test_loss = mape.loss(y_pred,y_true).mean()
                test_loss = criterion(y_pred,y_true)
                print(f'Epoch: {epoch}, Iter: {i}, Train Loss: {loss.detach().numpy()}, Test Loss: {test_loss.detach().numpy()}')
                test_losses.append(test_loss.detach().numpy())
                #error = np.abs(y_pred.detach().numpy() - y_true.detach().numpy())
                #m,h = mean_confidence_interval(error)
                #print('0.95 confidence: {} +- {}'.format(m,h))

    plt.plot(train_losses, label='Loss pri treniranju')
    plt.plot(test_losses, label='Loss pri validaciji')
    plt.xlabel('Iteracija')
    plt.ylabel('Vrednost Loss funkcije')
    print('Min test loss: {}'.format(min(test_losses)))
    plt.legend()
    plt.show()

    # train_data_df, train_y, test_data, test_y = data_from_csv(split=1.0)
    # train_data = torch.tensor(train_data_df.values, dtype=torch.float32)
    # train_data.requires_grad = True
    # y_pred = nn(train_data)

    # for i in range(0, train_data.shape[0]):
    #     y_pred[i].backward(retain_graph=True)
    
    # avg_grads = torch.mean(torch.abs(train_data.grad), dim=0).detach().numpy()
    # bars = plt.bar(range(avg_grads.shape[0]), avg_grads)
    # plt.bar_label(container=bars, labels=np.arange(0,len(avg_grads)))
    # plt.show()

    # cols_where_grad_small = [ train_data_df.columns[i] for i,x in enumerate(avg_grads) if x < 500]
    return None, nn, loss

def main():
    # small_grad_cols_cnt = dict()
    # for i in range(1):
    #     cols, _ = train()
    #     for col in cols:
    #         small_grad_cols_cnt[col] = small_grad_cols_cnt.get(col, 0) + 1
    # print(small_grad_cols_cnt)   
    _, nn, loss = train()
    
    test_data = pd.read_csv('test_processed.csv')
    test_data = test_data.reset_index(drop=True)
    y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32))
    y_pred = y_pred.detach().numpy().flatten()
    pred_df = pd.DataFrame(y_pred, columns=['SalePrice'])
    pred_df.index += 1461
    pred_df.to_csv('pred.csv')

if __name__ == '__main__':
    main()