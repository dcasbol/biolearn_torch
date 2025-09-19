import torch
from torchvision.datasets import CIFAR10
from biolayer import BioConv2d
from visualization import LayerVisualizer

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

bio_conv = BioConv2d(3, num_neurons, kernel_size, stride=3)
weights = bio_conv.weight[:, :3, :, :].permute((0, 2, 3, 1))
vis = LayerVisualizer(weights)

for weight in bio_conv.train(cifar_data, batch_size=batch_size, epsilon=learning_rate):
	vis.update()
