import scipy.io
import numpy as np
from torch.utils.data import Dataset
import torch

class MnistDataset(Dataset):

	def __init__(self):
		N  = 784
		Nc = 10
		mat = scipy.io.loadmat('mnist_all.mat')
		M=np.zeros((0,N))
		for i in range(Nc):
			M=np.concatenate((M, mat['train'+str(i)]), axis=0)
		M=M/255.0
		self._M = torch.tensor(M, dtype=torch.float, requires_grad=False)

	def __len__(self):
		return 60000

	def __getitem__(self, i):
		return self._M[i]