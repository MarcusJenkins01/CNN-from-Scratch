import numpy as np
from .tensor import Tensor
import random


class ConvolutionalLayer2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    
    # Initialise the filters and biases using normal distribution with std of 0.01
    self.filters = Tensor([
      [random.gauss(0, 0.01) for _ in range(in_channels * kernel_size**2)]
      for _ in range(out_channels)])
    self.biases = Tensor([random.gauss(0, 0.01) for _ in range(out_channels)])
  
  def im2col(self, x):
    """Returns each (kernel_size * kernel_size) patch for each channel in the input as a list (for matrix multiplication compatibility)"""
    batch_size, channels, height, width = x.get_shape()
    padded_height = height + 2 * self.padding
    padded_width = width + 2 * self.padding

    batch_columns = []
    for b in range(batch_size):
      columns = []
      
      # Create the padded input
      x_padded = Tensor.zeros((channels, padded_height, padded_width))
      for c in range(channels):
        for i in range(height):
          for j in range(width):
            x_padded.data[c][i + self.padding][j + self.padding] = x.data[b][c][i][j]

      # Create the columns from the padded input
      for i in range(0, padded_height - self.kernel_size + 1, self.stride):
        for j in range(0, padded_width - self.kernel_size + 1, self.stride):
          patch = []
          for c in range(channels):
            for ki in range(self.kernel_size):
              for kj in range(self.kernel_size):
                row = i + ki
                col = j + kj
                patch.append(x_padded.data[c][row][col])
          columns.append(patch)
      
      batch_columns.append(columns)

    # Return the columns as a Tensor
    return Tensor(batch_columns)
  
  def forward(self, x):
    print(x.get_shape())
    batch_size, channels, height, width = x.get_shape()
    im2col_matrix = self.im2col(x)

    # Prepare the output Tensor
    out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
    out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
    out = Tensor.zeros((batch_size, self.out_channels, out_height, out_width))

    # Perform convolution by matrix multiplication
    for bi in range(batch_size):
      batch_patches = im2col_matrix[bi].transpose()
      conv_results = self.filters.matmul(batch_patches)

      # Place results back into the output tensor
      for fi in range(self.out_channels):
        out.data[bi][fi] = conv_results[fi].reshape((out_height, out_width)).data

    return out
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)
