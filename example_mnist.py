import torch
import mnist
import matplotlib.pyplot as plt
from time import time
from torch.utils.data import DataLoader
import numpy as np
from biolayer import BioLinear
import scipy.io

Nc   = 10   # num. of classes
N    = 784  # Sample size
Nep  = 300  # Number of epochs
Num  = 100  # Batch size
eps0 = 2e-2 # Learning rate

Kx=5
Ky=5
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array

def draw_weights(synapses, Kx, Ky):
	yy=0
	HM=np.zeros((28*Ky,28*Kx))
	for y in range(Ky):
		for x in range(Kx):
			HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:].reshape(28,28)
			yy += 1
	plt.clf()
	nc=np.amax(np.absolute(HM))
	im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
	fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
	plt.axis('off')
	fig.canvas.draw()
	fig.show()

fig=plt.figure(figsize=(13,10))

mat = scipy.io.loadmat('mnist_all.mat')
M=np.zeros((0,N))
for i in range(Nc):
	M=np.concatenate((M, mat['train'+str(i)]), axis=0)
M = torch.tensor(M, dtype=torch.float)/255.0

bio_linear = BioLinear(N, hid)

for weight in bio_linear.train(M, Nep, batch_size=Num, epsilon=eps0):
	draw_weights(bio_linear.weight.data, Kx, Ky)
		
	