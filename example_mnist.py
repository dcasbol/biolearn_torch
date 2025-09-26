import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from training import train_layer

MNIST_DIR = '~/DataSets/'

num_inputs = 784
batch_size = 32
num_neurons = 5 * 5

M = MNIST(MNIST_DIR, download=True).data.detach().view(-1, 28 * 28).float()
M = (M - M.mean(0)) / 255.0

linear = nn.Linear(num_inputs, num_neurons, bias=False)
data_loader = DataLoader(M, batch_size=batch_size, shuffle=True, drop_last=True)
train_layer(linear, data_loader)
