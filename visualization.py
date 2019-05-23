import matplotlib.pyplot as plt
import numpy as np

class LayerVisualizer(object):
	"""
	Abstract class for the visualization of weights convergence.
	"""
	def _init_plot(self, figsize):
		raise NotImplementedError

	def _update(self):
		raise NotImplementedError

	def _update_img(self, monochrome=False, intra_norm=False):
		Sy, Sx = self._Sy, self._Sx
		yy = 0
		for i in range(self._n):
			y, x = i//self._Kx, i%self._Kx
			kernel = self._w[i].cpu().numpy()
			if intra_norm:
				kernel -= np.amin(kernel)
				kernel /= np.amax(kernel) + 1e-10
			self._vals[y*Sy:(y+1)*Sy,x*Sx:(x+1)*Sx,:]=kernel
		if not intra_norm and not monochrome:
			vmin = np.amin(self._vals)
			vmax = np.amax(self._vals)
			vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)
			self._vals /= 2*abs(vmax)
			self._vals += 0.5
			#self._vals -= np.amin(self._vals)
			#self._vals /= np.amax(self._vals) + 1e-10
		self._img.set_data(self._vals[:,:,0] if monochrome else self._vals)
		if monochrome:
			vmin, vmax = self._img.get_clim()
			vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)
			vmin = 0.9*vmin + 0.1*np.amin(self._vals)
			vmax = 0.9*vmax + 0.1*np.amax(self._vals)
			self._img.set_clim(vmin, vmax)
		self._fig.canvas.draw()

	def close(self):
		plt.close(self._fig)

class ConvLayerVisualizer(LayerVisualizer):
	"""
	This class helps visualize the kernels of a conv layer.
	Only first 3 input dimensions of each kernel are displayed.
	"""

	def __init__(self, conv_layer, figsize=(5,5), intra_kernel_norm=False):
		self._Sx, self._Sy = conv_layer.kernel_size
		self._n = conv_layer.out_channels
		self._Kx = int(np.round(np.sqrt(self._n)))
		self._Ky = int(np.ceil(self._n/self._Kx))
		self._w = conv_layer.weight[:,:3,:,:].detach()
		self._w = self._w.transpose(1,2).transpose(2,3)
		self._ikn = intra_kernel_norm
		self._init_plot(figsize)

	def _init_plot(self, figsize):
		self._fig = plt.figure(figsize=figsize)
		self._vals = np.zeros((self._Sy*self._Ky,self._Sx*self._Kx,3))
		self._img = plt.imshow(self._vals)
		plt.axis('off')
		self._fig.show()

	def update(self):
		self._update_img(intra_norm=self._ikn)

class LinearLayerVisualizer(LayerVisualizer):
	"""
	Visualize convergence of weights of a linear layer.
	"""

	def __init__(self, linear_layer, figsize=(5,5), as_heatmap=False, img_size=None):
		self._n = linear_layer.out_features
		self._d = linear_layer.in_features
		self._w = linear_layer.weight.detach()
		self._hm = as_heatmap
		if as_heatmap:
			h = self._w.shape[0]
			d = self._w.shape[1]
			if img_size is not None:
				self._Sy, self._Sx = img_size
				assert self._Sx*self._Sy == d, "img_size doesn't match dimension 1"
			else:
				self._Sy = int(np.sqrt(d))
				self._Sx = d//self._Sy
				assert self._Sx*self._Sy == self._w.shape[1], \
					"Can't show vector of size {} as a heatmap".format(d)
				self._Ky = int(np.sqrt(h))
				self._Kx = h//self._Ky
			self._w = self._w.view(h, self._Sy, self._Sx, 1)
		else:
			self._w = self._w.t()
		self._init_plot(figsize)

	def _init_plot(self, figsize):
		self._fig = plt.figure(figsize=figsize)
		if self._hm:
			self._vals = np.zeros((self._Sy*self._Ky, self._Sx*self._Kx, 1))
			self._img = plt.imshow(self._vals[:,:,0], cmap='bwr', vmin=-0.5, vmax=0.5)
			self._fig.colorbar(self._img)
			plt.axis('off')
		self._fig.show()

	def update(self):
		if self._hm:
			self._update_img(monochrome=True)
		else:
			self._fig.clf()
			w0 = self._w.numpy()
			plt.plot(np.sum(w0, axis=1))
			self._fig.canvas.draw()