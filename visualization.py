import pygame
import torch
import numpy as np
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

	def __post_init__(self):

		vspace = self.height

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
		self._screen = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption("U-Training")
		self._screen.fill(WHITE)

		font = pygame.font.SysFont(None, 48)
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
