import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from training import train_layer

CIFAR_DIR = '~/DataSets/'

batch_size = 50
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

conv = nn.Conv2d(3, num_neurons, kernel_size, stride=3, bias=False)
data_loader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, drop_last=True)
train_layer(conv, data_loader, n_random_patches=2 * num_neurons)
