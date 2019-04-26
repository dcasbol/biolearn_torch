import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

no_grad_tensor = lambda x: torch.tensor(x, dtype=torch.float, requires_grad=False)

class BioLinear(nn.Linear):

	def __init__(self, in_features, out_features, p=2.0, delta=0.4, k=2):
		super(BioLinear, self).__init__(in_features, out_features, bias=False)

		assert p >= 2, 'Lebesgue norm must be greater or equal than 2'
		assert k >= 2, 'ranking parameter must be greater or equal than 2'
		assert k <= out_features, "ranking parameter can't exceed number of hidden units"
		self._p = no_grad_tensor(p)
		self._delta = no_grad_tensor(delta)
		self._k = k
		self._prec = no_grad_tensor(1e-30)
		self.weight.data.uniform_(1e-5, 1) # Seems to work better

	def train_step(self, inputs, eps):
		with torch.no_grad():
			synapses = self.weight.data
			hid = synapses.shape[0]
			Num = inputs.shape[0]
			N   = inputs.shape[1]

			sig=synapses.sign()
			tot_input=torch.matmul(sig*synapses.abs().pow(self._p-1), inputs.t())

			idx_batch=torch.arange(inputs.shape[0])
			y=torch.argsort(tot_input,dim=0)
			yl=torch.zeros(hid, Num)
			yl[y[hid-1,:],idx_batch]=1.0
			yl[y[hid-self._k],idx_batch]=-self._delta
			
			xx=(yl*tot_input).sum(1)
			ds  = torch.matmul(yl,inputs) - xx.view(xx.shape[0],1).repeat(1,N)*synapses
			
			nc=ds.abs().max()
			if nc<self._prec:
				nc=self._prec
			synapses += eps*(ds/nc)

	def train(self, train_data, epochs, batch_size=100, epsilon=2e-2):
		assert type(train_data) == torch.Tensor, 'train_data has to be a torch.Tensor'

		dataset = TensorDataset(train_data)
		loader  = DataLoader(dataset,
			batch_size = batch_size,
			shuffle    = True
		)
		for nep in range(epochs):
			eps = epsilon*(1-nep/epochs)
			for inputs, in loader:
				self.train_step(inputs, eps)
			yield self.weight.data