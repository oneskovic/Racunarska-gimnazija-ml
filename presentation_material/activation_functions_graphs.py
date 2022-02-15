import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def plot_activation_fucn(func, title, i):
    plt.subplot(2,3,i)
    x = torch.tensor(np.linspace(-10, 10, 2000), dtype=torch.float32)
    y = func(x)

    plt.grid()
    plt.title(title)
    plt.plot(x, y)


names = ['Tanh', 'Sigmoid', 'LeakyReLU', 'ReLU', 'ELU', 'Softplus']
functions = [torch.tanh, torch.sigmoid, nn.LeakyReLU(0.05), nn.ReLU(), nn.ELU(), nn.Softplus()]
for i, func in enumerate(functions):
    plot_activation_fucn(func, names[i], i+1)

plt.show()