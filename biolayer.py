import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

no_grad_tensor = lambda x: torch.tensor(x, dtype=torch.float, requires_grad=False)

def _compute_step_linear(inputs, synapses, p, delta, k, eps):
	with torch.no_grad():
		prec = no_grad_tensor(1e-30)
		hid = synapses.shape[0]
		Num = inputs.shape[0]
		N   = inputs.shape[1]

		sig=synapses.sign()
		tot_input=torch.matmul(sig*synapses.abs().pow(p-1), inputs.t())

		idx_batch=torch.arange(Num)
		y=torch.argsort(tot_input,dim=0)
		yl=torch.zeros(hid, Num)
		yl[y[hid-1,:],idx_batch]=1.0
		yl[y[hid-k],idx_batch]=-delta
		
		xx=(yl*tot_input).sum(1)
		ds  = torch.matmul(yl,inputs) - xx.view(xx.shape[0],1).repeat(1,N)*synapses
		
		nc=ds.abs().max()
		if nc<prec:
			nc=prec
		return eps*(ds/nc)

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
		synapses = self.weight.data
		wdelta = _compute_step_linear(inputs, synapses, self._p, self._delta, self._k, eps)
		synapses += wdelta

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

class BioConv2d(BioLinear):
	def __init__(self, in_channels, out_channels, kernel_size,
		stride=1, padding=0, dilation=1,
		p=2.0, delta=0.4, k=2):
		in_features = in_channels*kernel_size*kernel_size
		super(BioConv2d, self).__init__(in_features, out_channels,
			p=p, delta=delta, k=k)
		self.weight.data.uniform_(1e-5, 1)
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation

	def train_step(self, inputs, eps):
		assert len(inputs.shape) == 4, 'Inputs must be images with shape [B,C,H,W]'
		synapses = self.weight.data
		with torch.no_grad():
			folds = F.unfold(inputs, self.kernel_size,
				dilation=self.dilation, padding=self.padding, stride=self.stride)
			folds = folds.transpose(2,1).contiguous().view(-1, self.in_features)
		wdelta = _compute_step_linear(folds, synapses, self._p, self._delta, self._k, eps)
		synapses += wdelta