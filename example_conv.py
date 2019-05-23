import torch
from torchvision.datasets import CIFAR10
from biolayer import BioConv2d
from visualization import ConvLayerVisualizer

CIFAR_DIR = '~/DataSets/'

Num  = 50   # Batch size
eps0 = 2e-3 # Learning rate
S    = 5    # kernel size

Kx=10
Ky=10
hid=Kx*Ky   # number of hidden units that are displayed in Ky by Kx array

cifar = CIFAR10(CIFAR_DIR)
with torch.no_grad():
	cifar_data = (torch.tensor(cifar.data.view(), dtype=torch.float) / 255.)
	cifar_data = cifar_data.transpose(1,3).transpose(2,3) # [B,H,W,C] --> [B,C,H,W]

bio_conv = BioConv2d(3, hid, S, stride=3)
vis = ConvLayerVisualizer(bio_conv, intra_kernel_norm=False)

try:
	for weight in bio_conv.train(cifar_data, batch_size=Num, epsilon=eps0):
		vis.update()
except KeyboardInterrupt:
	vis.close()
