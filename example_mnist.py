import torch
import matplotlib.pyplot as plt
from biolayer import BioLinear
from visualization import LinearLayerVisualizer
from torchvision.datasets import MNIST

MNIST_DIR = '~/DataSets/'

Nc   = 10   # num. of classes
N    = 784  # Sample size
Nep  = 300  # Number of epochs
Num  = 100  # Batch size
eps0 = 2e-2 # Learning rate

Kx=5
Ky=5
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array

M = MNIST(MNIST_DIR, download=True).data.detach().view(-1, 28*28).float()
M -= M.mean(0)

bio_linear = BioLinear(N, hid)
vis = LinearLayerVisualizer(bio_linear, as_heatmap=True)

try:
	for weight in bio_linear.train(M, Nep, batch_size=Num, epsilon=eps0):
		vis.update()
except KeyboardInterrupt:
	vis.close()
	