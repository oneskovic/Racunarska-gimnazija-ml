import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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

def objective(trial):
    train_data, train_y, test_data, test_y = data_from_csv(split=0.8)
    input_size = train_data.shape[1]

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 16, 256)
    layer_cnt = trial.suggest_int('layer_cnt', 1, 10)
    
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
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_data, train_y, test_data, test_y = data_from_csv(split=0.8)
    for epoch in range(120):
        # Shuffle training and test set
        perm = np.random.permutation(train_data.shape[0])
        train_data = train_data.iloc[perm]
        train_y = train_y.iloc[perm]
        perm2 = np.random.permutation(test_data.shape[0])
        test_data = test_data.iloc[perm2]
        test_y = test_y.iloc[perm2]

        losses = []
        i= 0
        while i < train_data.shape[0]:
            batch_x = train_data[i:min(i+batch_size, train_data.shape[0])]
            batch_y = train_y[i:min(i+batch_size, train_data.shape[0])]
            optimizer.zero_grad()
            y_pred = nn(torch.tensor(batch_x.values, dtype=torch.float32))
            y_true = torch.tensor(batch_y.values, dtype=torch.float32).reshape(y_pred.shape)
            loss = criterion(y_pred,y_true)
            losses.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            i += batch_size

        y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32)).flatten().detach().numpy()
        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(test_y.values, dtype=torch.float32).flatten()

        loss = criterion(y_pred,y_true)

        if np.isnan(loss.detach().numpy()):
            raise optuna.exceptions.TrialPruned()

        print('Epoch {}: Loss {}'.format(epoch, loss.detach().numpy()))
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss

def eval_net(hparams):
    train_data, train_y, test_data, test_y = data_from_csv(split=0.8)
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
    optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_losses = []
    val_losses = []
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

        train_loss = np.array(losses).mean()
        print(f'Epoch {epoch} Training Loss {train_loss}')
        print(f'Epoch {epoch} Validation Loss {loss.item()}')
        train_losses.append(train_loss)
        val_losses.append(loss.item())

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
    cols_where_grad_small = [ train_data_df.columns[i] for i,x in enumerate(avg_grads) if x < 500]

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    return nn, loss


def main():
    # Uncomment to optimize hparams
    # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(
    #     min_resource=30, max_resource=100, reduction_factor=3))
    study = pickle.load(open('study.pkl', 'rb'))
    study.optimize(objective, n_trials=500)
    pickle.dump(study, open('study1.pkl', 'wb+'))
    optuna.visualization.plot_parallel_coordinate(study)
    
    # Load hparams
    study = pickle.load(open('study1.pkl', 'rb'))
    # study.optimize(objective, n_trials=200)
    # pickle.dump(study, open('study.pkl', 'wb+'))

    hparams = study.best_params
    print(hparams)
    #nn, loss = eval_net(hparams)
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