
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('mnist_all.mat')
from time import time

Nc=10 # num. of classes
N=784 # Sample size
Ns=60000 # Num. samples in training set

# M <-- Training set (normalized)
M=np.zeros((0,N))
for i in range(Nc):
	M=np.concatenate((M, mat['train'+str(i)]), axis=0)
M=M/255.0

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

eps0=2e-2    # learning rate
Kx=5
Ky=5
hid=Kx*Ky    # number of hidden units that are displayed in Ky by Kx array
mu=0.0
sigma=1.0
Nep=200      # number of epochs
Num=100      # size of the minibatch
prec=1e-30
delta=0.4    # Strength of the anti-hebbian learning
p=2.0        # Lebesgue norm of the weights
k=2          # ranking parameter, must be integer that is bigger or equal than 2

fig=plt.figure(figsize=(6.5,5))

# Synapses initialization. Could be any (test with uniform)
#synapses = np.random.normal(mu, sigma, (hid, N))
synapses = np.random.uniform(-1.0, 1.0, (hid, N+1))

ones = np.ones((1, Num))

for nep in range(Nep):

	eps=eps0*(1-nep/Nep) # lr annealing
	M=M[np.random.permutation(Ns),:] # Shuffle training set (inefficient)
	print(eps)

	t0 = time()

	for i in range(Ns//Num):

		# I think this is the input current
		inputs=np.transpose(M[i*Num:(i+1)*Num,:]) # batch
		inputs=np.concatenate([inputs, ones])

		sig=np.sign(synapses)
		tot_input=np.dot(sig*np.absolute(synapses)**(p-1),inputs)
		
		# This is the g function. Could be made more efficient with argmax.
		idx_batch=np.arange(Num)
		y=np.argsort(tot_input,axis=0)
		yl=np.zeros((hid,Num))
		yl[y[hid-1,:],idx_batch]=1.0
		yl[y[hid-k],idx_batch]=-delta
		
		xx=np.sum(np.multiply(yl,tot_input),1)
		ds=np.dot(yl,np.transpose(inputs)) - np.multiply(np.tile(xx.reshape(xx.shape[0],1),(1,N+1)),synapses)
		
		nc=np.amax(np.absolute(ds))
		if nc<prec:
			nc=prec
		synapses += eps*np.true_divide(ds,nc)

	print(time()-t0) #0.42
		
	draw_weights(synapses[:,:-1], Kx, Ky)