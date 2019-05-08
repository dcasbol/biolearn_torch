import torch
import mnist
import matplotlib.pyplot as plt
from time import time, sleep
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import numpy as np
from biolayer import BioConv2d
import scipy.io

CIFAR_DIR = '~/DataSets/'

Nep  = 300  # Number of epochs
Num  = 50   # Batch size
eps0 = 1e-2 # Learning rate
S    = 5    # kernel size

Kx=10
Ky=10
hid=Kx*Ky   # number of hidden units that are displayed in Ky by Kx array

def draw_weights(synapses, Kx, Ky):
	yy=0
	HM=np.zeros((S*Ky,S*Kx,3))
	for y in range(Ky):
		for x in range(Kx):
			HM[y*S:(y+1)*S,x*S:(x+1)*S,:]=synapses[yy,:].reshape(S,S,3)
			yy += 1
	plt.clf()
	HM -= np.amin(HM)
	HM /= np.amax(HM)
	im=plt.imshow(HM)
	plt.axis('off')
	fig.canvas.draw()

fig=plt.figure(figsize=(6.5,5))
fig.show()

cifar = CIFAR10(CIFAR_DIR)
with torch.no_grad():
	cifar_data = torch.tensor(cifar.data.view()[:1000], dtype=torch.float) / 255.
	cifar_data = cifar_data.transpose(1,3).transpose(2,3)

bio_conv = BioConv2d(3, hid, S)

for weight in bio_conv.train(cifar_data, Nep, batch_size=Num, epsilon=eps0):
	weights = bio_conv.weight.data.detach().numpy()
	weights = weights.transpose([0,2,3,1]).reshape(hid, S*S*3)
	draw_weights(weights, Kx, Ky)
		
	