import pygame
import torch
import numpy as np
from math import sqrt, ceil
from dataclasses import dataclass


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
DARK_BLUE = (30, 90, 150)


@dataclass
class LayerVisualizer(object):
	weights: torch.Tensor
	layer_id: str = "No Layer ID Provided"
	width: int = 500
	height: int = 500

	@torch.no_grad()
	def __post_init__(self):

		vspace = self.height

		# Keep a view of weights which is more suitable for visualization
		assert len(self.weights.shape) in [2, 4]
		if len(self.weights.shape) == 2:
			num_neurons, num_inputs = self.weights.shape
			depth = 1
			self._in_height, self._in_width = best_fitting_size(num_inputs)
		else:
			num_neurons, depth, kernel_height, kernel_width = self.weights.shape
			num_inputs = kernel_height * kernel_width
			self._in_height = kernel_height
			self._in_width = kernel_width

		self._out_height, self._out_width = best_fitting_size(num_neurons)
		self.weights = self.weights.view(num_neurons, num_inputs, depth)
		self._num_neurons = num_neurons
		self._num_inputs = num_inputs
		self._depth = depth
		self._canvas = torch.zeros(
			self._out_height * self._out_width,
			self._in_height * self._in_width,
			1 if depth == 1 else 3,
		)

		# Initialize visualization window
		pygame.init()
		self._screen = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption("U-Training")
		self._screen.fill(WHITE)

		font = pygame.font.SysFont(None, 24)
		title_text = font.render(self.layer_id, True, BLACK)
		title_rect = title_text.get_rect(center=(self.width // 2, 50))
		self._screen.blit(title_text, title_rect)
		vspace -= 50 + 48 // 2

		button_font = pygame.font.SysFont(None, 36)
		button_width = 150
		self._button_rect = pygame.Rect(self.width // 2 - button_width // 2, self.height - 60, button_width, 50)
		self._button_text = button_font.render("Stop", True, WHITE)
		vspace -= 60 + 50 // 2
		self._vspace = vspace

	def update(self) -> bool:
		self._update_img()
		self._update_button()
		pygame.display.flip()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if self._button_rect.collidepoint(event.pos):
					return False
		return True

	@torch.no_grad()
	def _update_img(self):

		# Copy weight values over to canvas, where some empty spots can be left
		if self._depth > 3:
			sensitivity_levels = self.weights.view(self._num_neurons * self._num_inputs, self._depth).mean(dim=0)
			depth_idx = sensitivity_levels.argsort(stable=True, descending=True)[:3]
			self._canvas[:self._num_neurons, :self._num_inputs, :] = self.weights[:, :, depth_idx]
		else:
			depth = min(self._depth, 3)
			self._canvas[:self._num_neurons, :self._num_inputs, :depth] = self.weights[:, :, :depth]

		# Reshape canvas so we can show neurons' weights in a grid
		weights = self._canvas.view(self._out_height, self._out_width, self._in_height, self._in_width, -1)
		weights = weights.permute((0, 2, 1, 3, 4))
		if weights.shape[-1] == 1:
			weights = weights.tile((1, 1, 1, 1, 3))
		weights = weights.reshape(self._out_height * self._in_height, self._out_width * self._in_width, 3)
		weights = weights.cpu().numpy()

		vmax = np.amax(np.abs(weights))
		img = (weights + vmax) / (2 * vmax + 1e-10)
		img = (255 * img).astype(np.uint8)
		img = pygame.surfarray.make_surface(img)
		img_size = min(self.width, self._vspace) * 0.95
		img = pygame.transform.scale(img, (img_size, img_size))
		img = pygame.transform.rotate(img, -90)
		img = pygame.transform.flip(img, True, False)
		img_rect = img.get_rect(center=(self.width // 2, self.height // 2))
		self._screen.blit(img, img_rect)

	def _update_button(self):
		mouse_pos = pygame.mouse.get_pos()
		color = BLUE if self._button_rect.collidepoint(mouse_pos) else DARK_BLUE
		pygame.draw.rect(self._screen, color, self._button_rect)
		text_rect = self._button_text.get_rect(center=self._button_rect.center)
		self._screen.blit(self._button_text, text_rect)


def best_fitting_size(n: int) -> tuple[int, int]:
	min_remainder = 1_000_000
	width, height = 1, n
	for i in range(2, ceil(sqrt(n)) + 1):
		w = i
		h = ceil(n / w)
		r = (w * h) - n
		if r < min_remainder or (r == min_remainder and w > width):
			min_remainder = r
			width, height = w, h
	return width, height
