from .variable import Variable
from .tensor import Tensor
from.model import Model
import random


class ConvolutionalLayer2d(Model):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    
    self.filters = Tensor([
      [Variable(random.gauss(0, 0.01)) for _ in range(in_channels * kernel_size**2)]
      for _ in range(out_channels)
    ])
    self.biases = Tensor([Variable(random.gauss(0, 0.01)) for _ in range(out_channels)])

  def im2col(self, x):
    batch_size, channels, height, width = x.get_shape()
    padded_height = height + 2 * self.padding
    padded_width = width + 2 * self.padding

    x_padded = Tensor.zeros((batch_size, channels, padded_height, padded_width))
    for b in range(batch_size):
      for c in range(channels):
        for i in range(height):
          for j in range(width):
            x_padded.data[b][c][i + self.padding][j + self.padding] = x.data[b][c][i][j]

    columns = []
    for b in range(batch_size):
      batch_patches = []
      for i in range(0, padded_height - self.kernel_size + 1, self.stride):
        for j in range(0, padded_width - self.kernel_size + 1, self.stride):
          patch = [x_padded.data[b][c][i + ki][j + kj] for c in range(channels) for ki in range(self.kernel_size) for kj in range(self.kernel_size)]
          batch_patches.append(patch)
      columns.append(batch_patches)
      
    return Tensor(columns)

  def forward(self, x):
    batch_size, channels, height, width = x.get_shape()
    im2col_matrix = self.im2col(x)
    out = []
    
    # Prepare the output Tensor
    out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
    out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
    
    for batch in im2col_matrix.data:
      filtered = self.filters.matmul2d(Tensor(batch).transpose())
      biased = [f + b for f, b in zip(filtered, self.biases.data)]
      out.append(biased)
      
    return Tensor(out).reshape((batch_size, self.out_channels, out_height, out_width))

  def get_parameters(self):
    return self.filters.flatten() + self.biases.flatten()
