import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def generate_data(n_samples=1, n_inputs = 50):
    x = np.random.randn(n_samples,n_inputs)
    y = np.zeros((n_samples,1))
    # for i in range(n_samples):
    #     y[i] =  np.sin(x[i,0])*x[i,1]**2 + np.cos(x[i,2]**2)
    return x, y

class ForwardCollector():
    def __init__(self):
        pass

    def hook_fn(self, module, input, output):
        t = output.detach().numpy()
        self.stats = np.average(t, axis=0)
    
    def get_mean_std(self):
        return np.mean(self.stats), np.std(self.stats)

layers = []
collectors = []
input_size = 1000
neuron_cnt = 1000
layer_cnt = 11
for i in range(layer_cnt):
    if i == 0:
        l = nn.Linear(input_size, neuron_cnt)
    elif i == layer_cnt-1:
        l = nn.Linear(neuron_cnt, 1)
    else:
        l = nn.Linear(neuron_cnt,neuron_cnt)

    # torch.nn.init.kaiming_normal_(l.weight)
    torch.nn.init.normal_(l.weight, mean=0, std=1)
    # torch.nn.init.xavier_normal_(l.weight)
    torch.nn.init.constant_(l.bias, 0)
    layers.append(l)

    if i == layer_cnt-1:
        continue

    relu = nn.ReLU()
    collector = ForwardCollector()
    collectors.append(collector)
    relu.register_forward_hook(collector.hook_fn)
    layers.append(relu)
net = nn.Sequential(*layers)

optim = torch.optim.SGD(params=net.parameters(), lr=0.001)
loss = nn.MSELoss()
x, y = generate_data(n_inputs=input_size)

with torch.no_grad():
    y_pred = net(torch.from_numpy(x).float())
    loss_value = loss(y_pred, torch.from_numpy(y).float())
    print(loss_value)

plt.suptitle('Aktivacije neurona po slojevima za Ha inicijalizaciju')
for i in range(len(collectors)):
    plt.subplot(2,5,i+1)
    plt.title(f'Sloj {i}')
    
    plt.hist(collectors[i].stats, bins=50, range=(-1,1))
plt.subplots_adjust(hspace=0.5)
plt.show()

plt.cla()
mean_stats = []
std_stats = []
for i in range(len(collectors)):
    mean, std = collectors[i].get_mean_std()
    mean_stats.append(mean)
    std_stats.append(std)
plt.subplot(2,1,1)
plt.plot(mean_stats, label='Srednja vrednost aktivacija')
plt.legend()
plt.subplot(2,1,2)
plt.plot(std_stats, label='Standardna devijacija aktivacija')
plt.legend()
plt.show()
# train_losses = []
# val_losses = []
# for epoch in range(500):
#     y_pred = net(torch.from_numpy(x).float())
#     loss_value = loss(y_pred, torch.from_numpy(y).float())
#     optim.zero_grad()
#     loss_value.backward()
#     optim.step()
#     print('epoch: {}, train loss: {}'.format(epoch, loss_value.item()))
#     train_losses.append(loss_value.item())
#     testx, testy = generate_data(n_samples=500)
#     with torch.no_grad():
#         y_pred = net(torch.from_numpy(testx).float())
#         val_loss = loss(y_pred, torch.from_numpy(testy).float())
#         val_losses.append(val_loss.item())
#         print('epoch: {}, val loss: {}'.format(epoch, val_loss.item()))

# plt.plot(train_losses, label='train loss')
# plt.plot(val_losses, label='val loss')
# plt.legend()
# plt.show()
