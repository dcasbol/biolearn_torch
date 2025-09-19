import pygame
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class LayerVisualizer(object):
	weights: torch.Tensor
	window_width: int = 250

	def __post_init__(self):

		# Keep a view of weights which is more suitable for visualization
		weights = self.weights.detach()
		if len(weights.shape) == 2:
			num_neurons, num_inputs = weights.shape
			depth = 1
		else:
			num_neurons, kernel_height, kernel_width, depth = weights.shape
			num_inputs = kernel_height * kernel_width
		self._Sy = int(np.sqrt(num_inputs))
		self._Sx = num_inputs//self._Sy
		assert self._Sx * self._Sy == num_inputs, \
			f"Inputs and outputs should be square, got {num_inputs=}, {num_neurons=}"
		self._Ky = int(np.sqrt(num_neurons))
		self._Kx = num_neurons//self._Ky
		self.weights = weights.view(self._Ky, self._Kx, self._Sy, self._Sx, depth)

		# Initialize visualization window
		pygame.init()
		self._screen = pygame.display.set_mode((self.window_width, self.window_width))
		pygame.display.set_caption("Weights")

	def update(self):
		self._update_img()

	def _update_img(self, intra_norm=False):
		# Sy, Sx = self._Sy, self._Sx
		# for i in range(self._n):
		# 	y, x = i // self._Kx, i % self._Kx
		# 	kernel = self._w[i].cpu().numpy()
		# 	if intra_norm:
		# 		kernel -= np.amin(kernel)
		# 		kernel /= np.amax(kernel) + 1e-10
		# 	self._vals[y * Sy:(y + 1) * Sy, x * Sx:(x + 1) * Sx, :] = kernel
		weights = self.weights.permute((0, 2, 1, 3, 4))
		if weights.shape[-1] == 1:
			weights = weights.tile((1, 1, 1, 1, 3))
		weights = weights.reshape(self._Ky * self._Sy, self._Kx * self._Sx, 3)
		weights = weights.cpu().numpy()
		vmax = np.amax(np.abs(weights))
		img = (weights + vmax) / (2 * vmax + 1e-10)
		img = (255 * img).astype(np.uint8)
		img = pygame.surfarray.make_surface(img)
		img = pygame.transform.scale(img, (250, 250))
		img = pygame.transform.rotate(img, -90)
		img = pygame.transform.flip(img, True, False)
		self._screen.blit(img, (0, 0))
		pygame.display.flip()