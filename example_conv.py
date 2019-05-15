import torch
from torchvision.datasets import CIFAR10
from biolayer import BioConv2d
from visualization import ConvLayerVisualizer

CIFAR_DIR = '~/DataSets/'

Nep  = 300  # Number of epochs
Num  = 50   # Batch size
eps0 = 2e-3 # Learning rate
S    = 7    # kernel size

Kx=5
Ky=5
hid=Kx*Ky   # number of hidden units that are displayed in Ky by Kx array

cifar = CIFAR10(CIFAR_DIR)
with torch.no_grad():
	cifar_data = (torch.tensor(cifar.data.view(), dtype=torch.float) / 255.)
	cifar_data = cifar_data.transpose(1,3).transpose(2,3) # [B,H,W,C] --> [B,C,H,W]

bio_conv = BioConv2d(3, hid, S, stride=3)
vis = ConvLayerVisualizer(bio_conv, intra_kernel_norm=True)

try:
	for weight in bio_conv.train(cifar_data, Nep, batch_size=Num, epsilon=eps0):
		vis.update()
except KeyboardInterrupt:
	vis.close()
	