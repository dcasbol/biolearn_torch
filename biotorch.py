import torch
import mnist
import matplotlib.pyplot as plt
from time import time
from torch.utils.data import DataLoader
import numpy as np
from biolayer import BioLinear

Nc=10 # num. of classes
N=784 # Sample size
Ns=60000 # Num. samples in training set

no_grad_tensor = lambda x: torch.tensor(x, dtype=torch.float, requires_grad=False)

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

eps0=no_grad_tensor(2e-2)    # learning rate
prec=no_grad_tensor(1e-30)

Kx=5
Ky=5
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array

Nep=200      # number of epochs
Num=100      # size of the minibatch

delta=0.4    # Strength of the anti-hebbian learning
p=2.0        # Lebesgue norm of the weights
k=2          # ranking parameter, must be integer that is bigger or equal than 2

fig=plt.figure(figsize=(6.5,5))

mnist_data  = mnist.MnistDataset()
data_loader = DataLoader(mnist_data,
	batch_size = Num,
	shuffle    = True
)

bio_linear = BioLinear(N, hid)

for nep in range(Nep):

	eps=eps0*(1-nep/Nep) # lr annealing
	print(eps)

	t0 = time()

	for inputs in data_loader:
		bio_linear.bio_step(inputs, eps)

	print(time()-t0) #0.42
		
	draw_weights(bio_linear.weight.data, Kx, Ky)