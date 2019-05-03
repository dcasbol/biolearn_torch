import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from time import time

no_grad_tensor = lambda x: torch.tensor(x, dtype=torch.float, requires_grad=False)

def _compute_step_linear(inputs, synapses, p, delta, k, eps):
	with torch.no_grad():
		prec = no_grad_tensor(1e-30)
		hid = synapses.shape[0]
		Num = inputs.shape[0]
		N = inputs.shape[1]
		idx_batch = torch.arange(Num)

		sig=synapses.sign()
		tot_input=torch.matmul(sig*synapses.abs().pow(p-1), inputs.t())

		values = tot_input.clone()
		y1 = torch.argmax(values, 0)
		y = y1
		for i in range(k-1):
			values[y,idx_batch] = -1e10
			y = torch.argmax(values, 0)
		y2 = y

		yl=torch.zeros(hid, Num)
		yl[y1,idx_batch]=1.0
		yl[y2,idx_batch]=-delta
		
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

class BioConv2d(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size,
		stride=1, padding=0, dilation=1, p=2.0, delta=0.4, k=2):
		in_features = in_channels*kernel_size*kernel_size
		super(BioConv2d, self).__init__(in_channels, out_channels, kernel_size,
			stride=stride, padding=padding, dilation=dilation, bias=False)
		self.weight.data.uniform_(1e-5, 1)
		self._p = p
		self._delta = delta
		self._k = k
		self._in_features = in_features

	def train_step(self, inputs, eps):
		assert len(inputs.shape) == 4, 'Inputs must be images with shape [B,C,H,W]'
		synapses = self.weight.data
		
		with torch.no_grad():
			blocks = F.unfold(inputs, self.kernel_size,
				dilation=self.dilation, padding=self.padding, stride=self.stride)
			blocks = blocks.transpose(2,1).contiguous().view(-1, self._in_features)
			syn_flat = synapses.view(synapses.shape[0], -1)
		wdelta = _compute_step_linear(blocks, syn_flat, self._p, self._delta, self._k, eps)
		wdelta = wdelta.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
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

def _compute_step_conv(inputs, synapses, stride, padding, dilation, p, delta, k, eps):
	with torch.no_grad():
		prec = no_grad_tensor(1e-30)
		hid = synapses.shape[0]
		N   = synapses.shape[1]*synapses.shape[2]*synapses.shape[3] # inputs to neuron
		batch_size = inputs.shape[0]

		sig=synapses.sign()
		tot_input=F.conv2d(inputs, sig*synapses.abs().pow(p-1),
			stride=stride, padding=padding, dilation=dilation)
		# tot_input=torch.matmul(sig*synapses.abs().pow(p-1), inputs.t())
		# [H,D] x [D,N] --> [H,N]

		tot_input_flat=tot_input.transpose(0,1).contiguous().view(hid, -1)
		# tot_input [batch,hid,H2,W2] --> [hid,batch*H2*W2]
		Num = tot_input_flat.shape[1] # batch*H2*W2
		idx_batch=torch.arange(Num)
		values = tot_input_flat.clone()
		y1 = torch.argmax(values, 0)
		y = y1
		for i in range(k-1):
			values[y, idx_batch] = -1e10
			y = torch.argmax(values, 0)
		y2 = y
		yl=torch.zeros(hid, Num) # Por cada neurona, tantas veces como se ejecutó
		yl[y1, idx_batch]=1.0 # 1 a la que se activó más
		yl[y2, idx_batch]=-delta

		xx=(yl*tot_input_flat).sum(1) # [hid]
		# [hid,Num] x [Num,D] --> [hid,D] (acumula refuerzo por sinapsis)
		# Num ~ batch*H2*W2
		# D ~ C*k1*k2

		kernel_size = (synapses.shape[2], synapses.shape[3])
		blocks = F.unfold(inputs, kernel_size,
			stride=stride, padding=padding, dilation=dilation)
		blocks = blocks.transpose(2,1).contiguous().view(-1, N)
		flat_synapses = synapses.view(hid, N)
		ds = torch.matmul(yl,blocks) - xx.view(hid,1).repeat(1,N)*flat_synapses
		
		nc=ds.abs().max()
		if nc<prec:
			nc=prec
		return (eps*(ds/nc)).view(hid, -1, kernel_size[0], kernel_size[1])