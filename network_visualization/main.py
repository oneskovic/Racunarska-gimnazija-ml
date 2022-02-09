import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
from collections import namedtuple
from data_drawer import DataDrawer


def train_net(hparams, train_x, train_y, test_x, test_y, export_png=False):

    # For drawing
    x_draw = points = np.random.uniform(low=-2, high=2, size=(10000, 2))

    # Add the network layers
    input_size = train_x.shape[1]
    output_size = hparams.classes_cnt
    layer_cnt = len(hparams.layer_sizes)
    layers = []
    layers.append(nn.Linear(input_size, hparams.layer_sizes[0]))
    layers.append(hparams.activation_function())
    for i in range(layer_cnt-1):
        layers.append(nn.Linear(hparams.layer_sizes[i], hparams.layer_sizes[i+1]))
        layers.append(hparams.activation_function())
    layers.append(nn.Linear(hparams.layer_sizes[-1], output_size))

    # Initialize the net, loss function, and optimizer
    model = nn.Sequential(*layers)
    batch_size = hparams.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(hparams.epochs):
        # Train in minibatches
        avg_loss = 0.0
        for i in range(0, train_x.shape[0], batch_size):
            batch_x = torch.tensor(train_x[i:i+batch_size,:], dtype=torch.float32)
            batch_y = torch.tensor(train_y[i:i+batch_size], dtype=torch.long)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / train_x.shape[0]
        
            if i % 10 == 0:        
                x,y = predict_labels(model,points=x_draw)
                plot_labels(x,y, title='Epoch {} batch: {}'.format(epoch, i))
                if export_png:
                    plt.savefig(f'network_visualization/animation/predictions_epoch_{epoch:03}_batch_{i:05}.png')
                else:
                    plt.show(block=False)
                    plt.pause(0.01)
                plt.cla()

        print('Epoch: {}, Loss: {}'.format(epoch, avg_loss))
        # Get validation loss
        with torch.no_grad():
            output = model(torch.tensor(test_x, dtype=torch.float32))
            loss = loss_fn(output, torch.tensor(test_y, dtype=torch.long))
            print("Epoch: {}, Validation loss: {}".format(epoch, loss))

    return model

def classify_correct(points):
    p0 = np.array([0.8, 0.5])
    p1 = np.array([-0.5, 0.8])
    p2 = np.array([-0.5, 0.3])
    labels = np.zeros(points.shape[0])
    d0 = np.linalg.norm(points - p0, axis=1)
    d1 = np.linalg.norm(points - p1, axis=1)
    d2 = np.linalg.norm(points - p2, axis=1)
    sum_dist = d0 + d1 + d2
    labels[sum_dist<=3.0] = 1
    return labels

def generate_data(point_cnt = 1000):
    points = np.random.uniform(low=-2, high=2, size=(point_cnt, 2))
    labels = classify_correct(points)
    return points, labels

def predict_labels(model, point_cnt=1000, points=None):
    if points is None:
        points = np.random.uniform(low=-2, high=2, size=(point_cnt, 2))
    else:
        point_cnt = points.shape[0]
    with torch.no_grad():
        pred = model(torch.tensor(points, dtype=torch.float32))
        y_pred = torch.argmax(pred, dim=1).detach().numpy()
    
    return points, y_pred

def plot_labels(points, labels, title=None):
    point_cnt = points.shape[0]

    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive']
    for label in np.arange(0, labels.max()+1):
        points_in_set = points[labels == label]
        plt.scatter(points_in_set[:,0], points_in_set[:,1], label=f'Skup {label}', s=3, c=colors[label])

    if title is not None:
        plt.title(title)
    plt.legend(loc='upper right')

def main():
    # Generate data
    # points, labels = generate_data(2000)
    drawer = DataDrawer(2500)
    drawer.start()
    points, labels = drawer.get_data()
    classes_cnt = labels.max() + 1

    split_mask = np.random.choice([0,1], size=points.shape[0], p=[0.8, 0.2])
    train_x = points[split_mask==0]
    train_y = labels[split_mask==0]
    test_x = points[split_mask==1]
    test_y = labels[split_mask==1]

    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, s=3, cmap='viridis')
    plt.show()

    Hparams = namedtuple('Hparams', ['layer_sizes', 'batch_size', 'epochs', 'learning_rate', 'activation_function', 'classes_cnt'])
    hparams = Hparams(layer_sizes=[20, 20], batch_size=4, epochs=10, learning_rate=0.01, activation_function=torch.nn.ReLU, classes_cnt=classes_cnt)
    nn = train_net(hparams, train_x, train_y, test_x, test_y)

export_gif = False
main()

if export_gif:
    import imageio
    import glob
    filenames = glob.glob(pathname='network_visualization/animation/*.png')
    images = []
    for filename in sorted(filenames):
        images.append(imageio.imread(filename))
    imageio.mimsave('neuroni.gif', images, duration=0.2)