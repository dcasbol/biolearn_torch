import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from training import train_model

CIFAR_DIR = '~/DataSets/'

batch_size = 50
learning_rate = 2e-3
kernel_size = 5
num_neurons = 10 * 10

"""
Hebbian learning works best with positive inputs, which is also optimal when ReLU
activations are employed in subsequent layers. This is why inputs are normalized to the
range [0, 1] and not whitened.
"""
cifar = CIFAR10(CIFAR_DIR)
with torch.no_grad():
	cifar_data = torch.tensor(cifar.data.view(), dtype=torch.float) / 255
	cifar_data = cifar_data.transpose(1,3).transpose(2,3) # [B,H,W,C] --> [B,C,H,W]

data_loader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, drop_last=True)
conv_net = nn.Sequential(
	nn.Conv2d(3, 16, 3),
	nn.MaxPool2d(2, 2),
	nn.ReLU(),
	nn.Conv2d(16, 32, 3),
	nn.MaxPool2d(2, 2),
	nn.ReLU(),
	nn.Conv2d(32, 64, 3),
)

train_model(conv_net, data_loader)
