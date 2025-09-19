from biolayer import BioLinear
from visualization import LayerVisualizer
from torchvision.datasets import MNIST

MNIST_DIR = '~/DataSets/'

num_inputs = 784
batch_size = 32
learning_rate = 2e-3
num_neurons = 5 * 5

M = MNIST(MNIST_DIR, download=True).data.detach().view(-1, 28 * 28).float()
M = (M - M.mean(0)) / 255.0

bio_linear = BioLinear(num_inputs, num_neurons)
vis = LayerVisualizer(bio_linear.weight)

try:
	for weight in bio_linear.train(M, batch_size=batch_size, epsilon=learning_rate):
		vis.update()
except KeyboardInterrupt:
	pass
