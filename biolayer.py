import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from time import time
from pynput import keyboard

no_grad_tensor = lambda x: torch.tensor(x, dtype=torch.float, requires_grad=False)

def _compute_step_linear(inputs, synapses, p, delta, k, eps):
	with torch.no_grad():
		inputs /= inputs.norm(1) + 1e-10

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
		yl[y1,idx_batch]=1.
		yl[y2,idx_batch]=-delta*(tot_input[y2,idx_batch]>0).float()
		
		xx=(yl*tot_input).sum(1)
		ds  = torch.matmul(yl,inputs) - xx.view(xx.shape[0],1).repeat(1,N)*synapses
		
		nc=ds.abs().max() + prec
		return eps*(ds/nc)

class _BioBase(object):
	def __init__(self, p=2.0, delta=0.4, k=2):
		assert p >= 2, 'Lebesgue norm must be greater or equal than 2'
		assert k >= 2, 'ranking parameter must be greater or equal than 2'
		n_hiddens = getattr(self, 'out_features', None) or self.out_channels
		assert k <= n_hiddens, "ranking parameter can't exceed number of hidden units"
		self._p = no_grad_tensor(p)
		self._delta = no_grad_tensor(delta)
		self._k = k
		self.weight.data.uniform_(1e-5,0.5)
		w = self.weight.data
		with torch.no_grad():
			if len(self.weight) == 4:
				norm = w.view(w.shape[0], -1).norm(1).view(w.shape[0],1,1,1)
			else:
				norm = w.norm(1)
			w /= norm

	def train_step(self, inputs, eps):
		raise NotImplementedError

	def train(self, *args, **kwargs):
		ds = args[0]
		assert isinstance(ds, torch.Tensor) or isinstance(ds, DataLoader), 'Only dataset as Tensor or DataLoader allowed'
		if type(args[0]) == torch.Tensor:
			return self._train_from_tensor(*args, **kwargs)
		else:
			return self._train_from_dataloader(*args, **kwargs)

	def _train_from_tensor(self, train_data, epochs=None, batch_size=100, epsilon=2e-2):
		dataset = TensorDataset(train_data)
		loader  = DataLoader(dataset,
			batch_size = batch_size,
			shuffle    = True
		)
		return self._train_from_dataloader(loader, epochs, epsilon=epsilon)

	def _train_from_dataloader(self, loader, epochs=None, epsilon=2e-2):
		t0 = time()
		max_epochs = 300 if epochs is None else epochs
		nep = -1
		while epochs is None or nep < epochs:
			nep += 1
			ep = min(nep, max_epochs-1)
			eps = epsilon*(1-ep/max_epochs)
			for i, batch_samples in enumerate(loader):
				self.train_step(batch_samples[0], eps)
				if epochs is None and time()-t0 >= 0.25:
					t0 = time()
					yield self.weight.data

class BioLinear(_BioBase, nn.Linear):

	def __init__(self, in_features, out_features, **kwargs):
		nn.Linear.__init__(self, in_features, out_features, bias=False)
		_BioBase.__init__(self, **kwargs)

	def train_step(self, inputs, eps):
		synapses = self.weight.data
		wdelta = _compute_step_linear(inputs, synapses, self._p, self._delta, self._k, eps)
		synapses += wdelta

class BioConv2d(_BioBase, nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size,
		stride=1, padding=0, dilation=1, **kwargs):
		nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
			stride=stride, padding=padding, dilation=dilation, bias=False)
		_BioBase.__init__(self, **kwargs)
		self._in_features = in_channels*kernel_size*kernel_size
		self._output_shape = None

	def _extract_blocks(self, inputs):
		return F.unfold(inputs, self.kernel_size,
			dilation=self.dilation, padding=self.padding, stride=self.stride)

	def train_step(self, inputs, eps):
		assert len(inputs.shape) == 4, 'Inputs must be images with shape [B,C,H,W], {}'.format(inputs.shape)
		synapses = self.weight.data
		
		with torch.no_grad():
			blocks = self._extract_blocks(inputs)

			"""
			TODO include random patch selection?
			It seems that it generates more diverse patches, as there is
			high redundancy intra-image.
			"""
			random_patches = False
			if random_patches:
				perm = torch.randperm(blocks.size(2))
				idx = perm[:5]
				blocks = blocks[:,:,idx]

			blocks = blocks.transpose(2,1).contiguous().view(-1, self._in_features)
			syn_flat = synapses.view(synapses.shape[0], -1)

		wdelta = _compute_step_linear(blocks, syn_flat, self._p, self._delta, self._k, eps)

		with torch.no_grad():
			syn_flat += wdelta

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