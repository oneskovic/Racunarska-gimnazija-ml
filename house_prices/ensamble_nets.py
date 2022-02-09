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
import os
import glob
from sklearn.manifold import TSNE

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

class WeightedRMSLELoss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        
    def forward(self, pred, actual):
        squared_error = torch.pow((torch.log(pred + 1) - torch.log(actual + 1)),2)
        # squared_error = torch.mul(squared_error, self.weights)
        return torch.sqrt(torch.mean(squared_error))


def data_from_csv(split, csv_path='data_pca.csv'):
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

class EnsambleNet():
    def __init__(self):
        x,y,_,_ = data_from_csv(1.0)
        total_input_size = x.shape[0]

        # Initialize dataframe with column 'input_weight'
        self.input_weights = pd.DataFrame(np.ones((total_input_size, 1)), columns=['input_weight'])
        self.nets = []
        self.add_net_to_ensamble()

    def data_from_csv(self, is_director = False, split=0.8, csv_path='data_pca.csv'):
        train_x, train_y, test_x, test_y = data_from_csv(split, csv_path)
        if is_director:
            predictions_train = []
            predictions_test = []
            for i, net in enumerate(self.nets):
                pred = net(torch.tensor(train_x.values, dtype=torch.float32)).flatten().detach().numpy()
                pred -= np.mean(pred)
                pred /= np.std(pred)
                predictions_train.append(pred)

                pred = net(torch.tensor(test_x.values, dtype=torch.float32)).flatten().detach().numpy()
                pred -= np.mean(pred)
                pred /= np.std(pred)
                predictions_test.append(pred)

            for i in range(len(predictions_test)):
                train_x[f'pred_{i}'] = predictions_train[i]
                test_x[f'pred_{i}'] = predictions_test[i]
        else:
            mask = self.input_weights['input_weight'] > 1e-5
            train_index = mask[mask.index.intersection(train_x.index)].index
            train_x = train_x.loc[train_index]
            train_y = train_y.loc[train_index]
            test_index = mask[mask.index.intersection(test_x.index)].index
            test_x = test_x.loc[test_index]
            test_y = test_y.loc[test_index]
        return train_x, train_y, test_x, test_y

    def train_net(self, params, trial = None, is_director = False):
        print(f'Training net, is director: {is_director}')

        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        layer_cnt = len(params['layer_sizes'])
        layer_sizes = params['layer_sizes']
        train_data, train_y, test_data, test_y = self.data_from_csv(is_director)
        input_size = train_data.shape[1]

        # Set layer sizes
        layers = []
        layers.append(torch.nn.Linear(input_size, layer_sizes[0]))
        layers.append(torch.nn.ReLU())
        for i in range(1,layer_cnt):
            layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(torch.nn.ReLU())
        if is_director:
            # Classification
            # layers.append(torch.nn.Linear(layer_sizes[-1], len(self.nets)))
            layers.append(torch.nn.Linear(layer_sizes[-1], 1))
        else:
            layers.append(torch.nn.Linear(layer_sizes[-1], 1))

        # Initialize net
        nn = torch.nn.Sequential(*layers)
        
        # Train the net
        criterion = RMSLELoss()
        optimizer = torch.optim.Adam(nn.parameters(), lr=learning_rate, weight_decay=1e-5)
        for epoch in range(100):
            losses = []
            train_data, train_y, test_data, test_y = self.data_from_csv(is_director)

            if is_director:
                pass
                # Classification
                # preds = np.array([train_data['pred_' + str(i)].values for i in range(len(self.nets))])
                # diffs = np.abs(preds - np.array([train_y.values]*len(self.nets)))
                # train_optimal_choice = np.argmin(diffs, axis=0)


                # Temp
                # coords_for_label = dict()
                # for net_index in range(len(self.nets)):
                #     indices_where_net = np.where(train_optimal_choice == net_index)[0]
                #     x = train_data.iloc[indices_where_net].drop(columns=['pred_' + str(i)for i in range(len(self.nets))]).values
                #     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                #     tsne_results = tsne.fit_transform(x)
                #     coords_for_label[net_index] = tsne_results

                # for net_index in range(len(self.nets)):
                #     x = coords_for_label[net_index][:,0]
                #     y = coords_for_label[net_index][:,1]
                #     plt.scatter(x,y, label=f'Net {net_index}')
                # plt.legend()
                # plt.show()
                # Temp

                # Classification
                # preds = np.array([test_data['pred_' + str(i)].values for i in range(len(self.nets))])
                # diffs = np.abs(preds - np.array([test_y.values]*len(self.nets)))
                # test_optimal_choice = np.argmin(diffs, axis=0)

            # Select the weights for the selected inputs
            input_weights = self.input_weights.iloc[train_data.index,:]

            for i in range(0, train_data.shape[0], batch_size):
                optimizer.zero_grad()
                y_pred = nn(torch.tensor(train_data.iloc[i:i+batch_size, :].values, dtype=torch.float32))

                
                if is_director:
                    # Classification
                    # criterion = torch.nn.CrossEntropyLoss()
                    # y_true = torch.tensor(train_optimal_choice[i:i+batch_size], dtype=torch.long)
                    criterion = RMSLELoss()
                    y_true = torch.tensor(train_y.iloc[i:i+batch_size].values, dtype=torch.float32).reshape(y_pred.shape)
                else:
                    weights = torch.tensor(input_weights.iloc[i:i+batch_size].values, dtype=torch.float32).reshape(y_pred.shape)
                    criterion = WeightedRMSLELoss(weights)
                    y_true = torch.tensor(train_y.iloc[i:i+batch_size].values, dtype=torch.float32).reshape(y_pred.shape)

                loss = criterion(y_pred,y_true)
                losses.append(loss.detach().numpy())
                loss.backward()
                optimizer.step()



            if is_director:
                # Classification
                # y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32))
                # y_true = torch.tensor(test_optimal_choice, dtype=torch.long)
                y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32)).flatten().detach().numpy()
                y_pred = torch.tensor(y_pred)
                y_true = torch.tensor(test_y.values, dtype=torch.float32).flatten()
            else:
                y_pred = nn(torch.tensor(test_data.values, dtype=torch.float32)).flatten().detach().numpy()
                y_pred = torch.tensor(y_pred)
                y_true = torch.tensor(test_y.values, dtype=torch.float32).flatten()

            loss = criterion(y_pred,y_true)

            if np.isnan(loss.detach().numpy()):
                return nn, np.inf

            print('Epoch {}: Loss {}'.format(epoch, loss.detach().numpy()))
            if trial is not None:
                trial.report(loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return nn, loss

    def objective_director(self, trial):
        train_data, train_y, test_data, test_y = self.data_from_csv(is_director=True)
        input_size = train_data.shape[1]

        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        batch_size = 4
        layer_cnt = 3
        
        layer_sizes = []
        for i in range(layer_cnt):
            layer_sizes.append(trial.suggest_int(f'layer_size_{i}', 1, 2*input_size))

        params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'layer_sizes': layer_sizes
        }

        print(f'Params: {trial.params}')
        nn, loss = self.train_net(params, trial, True)
        pickle.dump(nn, open(f'trained_nets/temp_director_nets/nn_{trial.number}.pkl', 'wb+'))
        return loss

    def objective_actor(self, trial):
        train_data, train_y, test_data, test_y = self.data_from_csv(is_director=False)
        input_size = train_data.shape[1]

        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        batch_size = 4
        layer_cnt = 3
        
        layer_sizes = []
        for i in range(layer_cnt):
            layer_sizes.append(trial.suggest_int(f'layer_size_{i}', 1, 2*input_size))

        params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'layer_sizes': layer_sizes
        }

        print(f'Params: {trial.params}')
        nn, loss = self.train_net(params, trial, False)
        pickle.dump(nn, open(f'trained_nets/temp_actor_nets/nn_{trial.number}.pkl', 'wb+'))
        return loss

    def evaluate_director(self):
        x,y,_,_ = self.data_from_csv(is_director=True, split=1.0)
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        y_pred = self.director(torch.tensor(x.values, dtype=torch.float32)).flatten().detach().numpy()
        diff = np.abs(y.values - y_pred)
        mask1 = diff < 2500
        diff[mask1] = 1e-5
        self.input_weights['input_weight'] = diff / diff.max()


    def train_director(self):
        print('Training director ...')
        for f in glob.glob('trained_nets/temp_director_nets/*'):
            os.remove(f)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(
        min_resource=10, max_resource=100, reduction_factor=3))
        study.optimize(self.objective_director, n_trials=50)
        with open(f'trained_nets/temp_director_nets/nn_{study.best_trial.number}.pkl', "rb") as model:
            self.director = pickle.load(model)
        self.evaluate_director()

    def add_net_to_ensamble(self):
        print('Adding net to ensamble ...')
        for f in glob.glob('trained_nets/temp_actor_nets/*'):
            os.remove(f)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(
        min_resource=10, max_resource=100, reduction_factor=3))
        study.optimize(self.objective_actor, n_trials=25)
        with open(f'trained_nets/temp_actor_nets/nn_{study.best_trial.number}.pkl', "rb") as model:
            self.nets.append(pickle.load(model))
        self.train_director()

def main():

    nets = EnsambleNet()
    for _ in range(2):
        print(f'Nonzero weight count: {(nets.input_weights>1e-5).sum()}')
        # plt.hist(nets.input_weights, bins=50)
        # plt.show()
        nets.add_net_to_ensamble()
    pickle.dump(nets, open('trained_nets/ensamble_net2.pkl', 'wb+'))

    nets = pickle.load(open('trained_nets/ensamble_net2.pkl', 'rb'))
    nets.train_director()
    plt.hist(nets.input_weights, bins=50)
    plt.show()

    # Uncomment to optimize hparams
    # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(
    #     min_resource=30, max_resource=100, reduction_factor=3))
    # study.optimize(objective, n_trials=500)
    # pickle.dump(study, open('study4.pkl', 'wb+'))
    # optuna.visualization.plot_parallel_coordinate(study)
    
    # Load hparams
    # study = pickle.load(open('study4.pkl', 'rb'))
    # study.optimize(objective, n_trials=200)
    # pickle.dump(study, open('study.pkl', 'wb+'))

    # hparams = study.best_params
    # print(hparams)
    # nn, loss = eval_net(hparams)
    # trial_scores = np.array([study.trials[i].value if study.trials[i].value is not None else np.inf for i in range(len(study.trials))])
    # sorted_indices = np.argsort(trial_scores)
    # for i in range(10):
    #     value = study.trials[sorted_indices[i]].value
    #     params = study.trials[sorted_indices[i]].params
    #     print('Study {}: {}'.format(i, value))
    #     print(params)

if __name__ == '__main__':
    main()