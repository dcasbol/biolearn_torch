import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, Iterator, TypeVar
from visualization import LayerVisualizer

TrainableLayer = TypeVar("TrainableLayer", nn.Linear, nn.Conv2d)


@torch.no_grad()
def train_model(
	model: nn.Sequential,
	training_data: Iterable,
	input_pipeline: nn.Sequential = None,
	**kwargs,
) -> nn.Sequential:

	all_layers: list[nn.Module] = [] if input_pipeline is None else [l for l in input_pipeline]
	trained_layers: list[nn.Module] = []

	for layer in model:
		input_pipeline = nn.Sequential(*all_layers) if len(all_layers) > 0 else None
		if isinstance(layer, nn.Sequential):
			layer = train_model(layer, training_data, input_pipeline=input_pipeline, **kwargs)
		elif isinstance(layer, (nn.Linear, nn.Conv2d)):
			train_layer(layer, training_data, input_pipeline=input_pipeline, **kwargs)
		elif len(list(layer.parameters())) > 0:
			print(f"WARNING. Layer {layer} has parameters but won't be trained.")
		trained_layers.append(layer)
		all_layers.append(layer)

	return nn.Sequential(*trained_layers)


@torch.no_grad()
def train_layer(
	layer: TrainableLayer,
	training_data: Iterable,
	input_pipeline: nn.Sequential = None,
	n_random_patches: int = 0,
	**train_step_kwargs,
) -> TrainableLayer:
	assert isinstance(layer, (nn.Linear, nn.Conv2d)), f"Can't train {layer}"
	vis = LayerVisualizer(layer.weight, layer_id=str(layer))

	# Custom initialization
	layer.weight.uniform_(1e-5, 1.0)
	if layer.bias is not None:
		layer.bias.uniform_(1e-5, 1.0)

	data_iter = endless_iter(training_data)
	while True:
		if not vis.update():
			break
		input_data = next(data_iter)
		if input_pipeline is not None:
			input_data = input_pipeline(input_data)

		if isinstance(layer, nn.Conv2d):
			kh, kw = layer.kernel_size
			in_features = layer.in_channels * kh * kw
			blocks = F.unfold(
				input_data, layer.kernel_size, dilation=layer.dilation, padding=layer.padding, stride=layer.stride,
			)
			if n_random_patches > 0:
				perm = torch.randperm(blocks.size(2))
				idx = perm[:n_random_patches]
				blocks = blocks[:, :, idx]
			input_data = blocks.transpose(2, 1).contiguous().view(-1, in_features)

		synapses = layer.weight.view(layer.weight.size(0), -1)
		if layer.bias is not None:
			synapses = torch.cat([synapses, layer.bias.view(-1, 1)], dim=1)
			input_data = torch.cat([input_data, torch.ones(input_data.size(0), 1)], dim=1)

		syn_delta = train_step(synapses, input_data, **train_step_kwargs)

		if layer.bias is not None:
			synapses = layer.weight.view(layer.weight.size(0), -1)
			synapses += syn_delta[:, :-1]
			layer.bias += syn_delta[:, -1]
		else:
			synapses += syn_delta

	return layer


@torch.no_grad()
def train_step(
	synapses: Tensor,
	inputs: Tensor,
	lebesgue_norm: float = 2.0,
	penalization_factor: float = 0.4,
	competing_units: int = None,
	learning_rate: float = 2e-2,
) -> Tensor:
	assert lebesgue_norm >= 2, 'Lebesgue norm p must be greater or equal than 2'
	if competing_units is not None:
		assert competing_units >= 2, 'ranking parameter k must be greater or equal than 2'
	inputs = inputs / (inputs.norm(dim=1, keepdim=True) + 1e-10)

	num_hidden = synapses.shape[0]
	competing_units = max(2, competing_units or round(0.25 * num_hidden))
	assert competing_units <= num_hidden, "ranking parameter k can't exceed number of hidden units"
	batch_size, num_inputs = inputs.shape[0:2]
	idx_batch = torch.arange(batch_size)

	sig = synapses.sign()
	tot_input = torch.matmul(sig * synapses.abs().pow(lebesgue_norm - 1), inputs.t())

	values = tot_input.clone()
	y1 = torch.argmax(values, 0)
	y = y1
	for i in range(competing_units - 1):
		values[y, idx_batch] = -1e10
		y = torch.argmax(values, 0)
	y2 = y

	yl = torch.zeros(num_hidden, batch_size)
	yl[y1, idx_batch] = 1
	yl[y2, idx_batch] = -penalization_factor * (tot_input[y2, idx_batch] > 0).float()

	xx = (yl * tot_input).sum(1)
	ds = torch.matmul(yl, inputs) - xx.view(xx.shape[0], 1).repeat(1, num_inputs) * synapses

	nc = ds.abs().max() + 1e-30
	syn_delta = learning_rate * (ds / nc)
	return syn_delta


def endless_iter(iterable: Iterable) -> Iterator:
	while True:
		for item in iterable:
			yield item
