import torch
import matplotlib.pyplot as plt
from biolayer import BioLinear
from visualization import LinearLayerVisualizer
from torchvision.datasets import MNIST
import torch.nn.functional as F

MNIST_DIR = '~/DataSets/'

Nc   = 10   # num. of classes
N    = 784  # Sample size
Nep  = 300  # Number of epochs
Num  = 100  # Batch size
eps0 = 2e-3 # Learning rate

Kx=5
Ky=5
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array

M = MNIST(MNIST_DIR, download=True).data.detach().view(-1, 28*28).float()
M -= M.mean(0, keepdim=True)
M = F.pad(M, (0,1), value=1)

bio_linear = BioLinear(N+1, hid)
vis = LinearLayerVisualizer(bio_linear, as_heatmap=True)

try:
	for weight in bio_linear.train(M, batch_size=Num, epsilon=eps0):
		vis.update()
except KeyboardInterrupt:
	vis.close()
	