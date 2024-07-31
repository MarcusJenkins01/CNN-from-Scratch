from tensor import Tensor
import random

class ConvolutionalLayer2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
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
    """Returns each (kernel_size * kernel_size) patch for each channel in the input as a list (for dot product compatibility)"""
    batch_size, channels, height, width = x.get_shape()
    padded_height = height + 2 * self.padding
    padded_width = width + 2 * self.padding

    batch_columns = []
    for b in range(batch_size):
      columns = []
      
      # Create the padded input
      x_padded = [[[0.] * padded_width for _ in range(padded_height)] for _ in range(channels)]
      for c in range(channels):
        for i in range(height):
          for j in range(width):
            x_padded[c][i + self.padding][j + self.padding] = x.data[b][c][i][j]

      # Create the columns from the padded input
      for i in range(0, padded_height - self.kernel_size + 1, self.stride):
        for j in range(0, padded_width - self.kernel_size + 1, self.stride):
          patch = []
          for c in range(channels):
            for ki in range(self.kernel_size):
              for kj in range(self.kernel_size):
                row = i + ki
                col = j + kj
                patch.append(x_padded[c][row][col])
          columns.append(patch)
      
      batch_columns.append(columns)

    # Return the columns as a Tensor
    return Tensor(batch_columns)
  
  def forward(self, x):
    batch_size, channels, height, width = x.get_shape()
    im2col_matrix = self.im2col(x)
    
    # Create destination array
    out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
    out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
    out = [[[0.] for _ in range(self.out_channels)] for _ in range(batch_size)]

    # Apply the dot product between the image/feature map patch and the filter kernel
    for bi, columns in enumerate(im2col_matrix):
      for col in columns:
        for fi, _filter in enumerate(self.filters):
          out[bi][fi].append(_filter.dot(col))  # Append dot product to the corresponding output channel

    out = Tensor(out)
    return out.reshape((self.out_channels, out_height, out_width))
  
  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)


if __name__ == "__main__":
  conv = ConvolutionalLayer2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
  image = Tensor([random.gauss(0, 0.01) for _ in range(4 * 3 * 32 * 32)])
  image = image.reshape((4, 3, 32, 32))
  out = conv(image)
  print(out)
  