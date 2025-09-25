import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from time import time

no_grad_tensor = lambda x: torch.tensor(x, dtype=torch.float, requires_grad=False)


def _compute_step_linear(inputs, synapses, p, delta, k, eps):
	with torch.no_grad():
		inputs = inputs / (inputs.norm(dim=1, keepdim=True) + 1e-10)

		prec = no_grad_tensor(1e-30)
		num_hidden = synapses.shape[0]
		batch_size, num_inputs = inputs.shape[0:2]
		idx_batch = torch.arange(batch_size)

		sig = synapses.sign()
		tot_input = torch.matmul(sig * synapses.abs().pow(p-1), inputs.t())

		values = tot_input.clone()
		y1 = torch.argmax(values, 0)
		y = y1
		for i in range(k-1):
			values[y, idx_batch] = -1e10
			y = torch.argmax(values, 0)
		y2 = y

		yl = torch.zeros(num_hidden, batch_size)
		yl[y1, idx_batch]= 1
		yl[y2, idx_batch]= -delta * (tot_input[y2, idx_batch] > 0).float()
		
		xx = (yl * tot_input).sum(1)
		ds  = torch.matmul(yl, inputs) - xx.view(xx.shape[0], 1).repeat(1, num_inputs) * synapses
		
		nc = ds.abs().max() + prec
		return eps * (ds / nc)


class _BioBase:

	def __init__(self, *args, p=2.0, delta=0.4, k=2, **kwargs):
		super().__init__(*args, **kwargs)
		assert p >= 2, 'Lebesgue norm p must be greater or equal than 2'
		assert k >= 2, 'ranking parameter k must be greater or equal than 2'
		n_hiddens = getattr(self, 'out_features', None) or self.out_channels
		assert k <= n_hiddens, "ranking parameter k can't exceed number of hidden units"
		self._p = no_grad_tensor(p)
		self._delta = no_grad_tensor(delta)
		self._k = k

	def reset_parameters(self):
		# I don't know yet why +uniform init works better, but it does.
		self.weight.data.uniform_(1e-5, 1.0)
		if self.bias is not None:
			self.bias.data.uniform_(1e-5, 1.0)

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
		loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
		return self._train_from_dataloader(loader, epochs, epsilon=epsilon)

	def _train_from_dataloader(self, loader, epochs=None, epsilon=2e-2):
		t0 = time()
		max_epochs = 300 if epochs is None else epochs
		nep = -1
		while epochs is None or nep < epochs:
			nep += 1
			for i, batch_samples in enumerate(loader):
				self.train_step(batch_samples[0], epsilon)
				if epochs is None and time()-t0 >= 0.25:
					t0 = time()
					yield self.weight.data


class BioLinear(_BioBase, nn.Linear):

	def __init__(self, *args, bias=False, **kwargs):
		assert not bias, "Linear layers currently work only without bias."
		super().__init__(*args, **kwargs)

	def train_step(self, inputs, eps):
		synapses = self.weight.data
		wdelta = _compute_step_linear(inputs, synapses, self._p, self._delta, self._k, eps)
		synapses += wdelta
		# synapses *= (1 - 1e-5)  # TODO: add weight decay


class BioConv2d(_BioBase, nn.Conv2d):

	def __init__(self, *args, n_random_patches: int = 0, **kwargs):
		super().__init__(*args, **kwargs)
		in_channels, kernel_height, kernel_width = self.weight.shape[1:]
		self._in_features = in_channels * kernel_height * kernel_width
		self._n_random_patches = n_random_patches

	def _extract_blocks(self, inputs):
		return F.unfold(inputs, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

	def train_step(self, inputs, eps):
		dims = len(inputs.shape)
		assert dims == 4, f'Inputs must be images with shape [B,C,H,W], but a tensor of {dims} dimensions was given.'
		synapses = self.weight.data
		
		with torch.no_grad():
			blocks = self._extract_blocks(inputs)

			# Might generate more diverse patches, as there is high intra-image redundancy.
			if self._n_random_patches > 0:
				perm = torch.randperm(blocks.size(2))
				idx = perm[:self._n_random_patches]
				blocks = blocks[:,:,idx]

			blocks = blocks.transpose(2,1).contiguous().view(-1, self._in_features)
			syn_flat = synapses.view(synapses.shape[0], -1)
			if self.bias is not None:
				blocks = torch.cat([blocks, torch.ones(blocks.size(0), 1)], dim=1)
				syn_flat = torch.cat([syn_flat, self.bias.view(-1, 1)], dim=1)

			wdelta = _compute_step_linear(blocks, syn_flat, self._p, self._delta, self._k, eps)

			if self.bias is not None:
				syn_flat = synapses.view(synapses.shape[0], -1)
				syn_flat += wdelta[:,:-1]
				self.bias.data += wdelta[:,-1]
			else:
				syn_flat += wdelta


def _compute_step_conv(inputs, synapses, stride, padding, dilation, p, delta, k, eps):
	# TODO: (in construction) compute the learning step without the flattening of patches and kernels.
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
		yl=torch.zeros(hid, Num) # For each neuron, as many times as it was ran
		yl[y1, idx_batch]=1.0 # 1 to the one with highest activation
		yl[y2, idx_batch]=-delta

		xx=(yl*tot_input_flat).sum(1) # [hid]
		# [hid,Num] x [Num,D] --> [hid,D] (accumulates reinforcement per-synapsis)
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