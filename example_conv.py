import torch
import mnist
import matplotlib.pyplot as plt
from time import time
from torch.utils.data import DataLoader
import numpy as np
from biolayer import BioConv2d
import scipy.io

Nc   = 10   # num. of classes
N    = 784  # Sample size
Nep  = 300  # Number of epochs
Num  = 10  # Batch size
eps0 = 2e-2 # Learning rate

Kx=5
Ky=5
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array

def draw_weights(synapses, Kx, Ky):
	yy=0
	S = 10
	HM=np.zeros((S*Ky,S*Kx))
	for y in range(Ky):
		for x in range(Kx):
			HM[y*S:(y+1)*S,x*S:(x+1)*S]=synapses[yy,:].reshape(S,S)
			yy += 1
	plt.clf()
	nc=np.amax(np.absolute(HM))
	im=plt.imshow(HM,cmap='bwr',vmin=-nc,vmax=nc)
	fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
	plt.axis('off')
	fig.canvas.draw()
	fig.show()

fig=plt.figure(figsize=(6.5,5))

mat = scipy.io.loadmat('mnist_all.mat')
M=np.zeros((0,N))
for i in range(Nc):
	M=np.concatenate((M, mat['train'+str(i)]), axis=0)
M = torch.tensor(M, dtype=torch.float).view(-1,1, 28,28)/255.0

bio_linear = BioConv2d(1, hid, 10)

for weight in bio_linear.train(M, Nep, batch_size=Num, epsilon=eps0):
	draw_weights(bio_linear.weight.data.view(hid,10*10), Kx, Ky)
		
	