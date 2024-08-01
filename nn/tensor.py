from .variable import Variable
import numpy as np


class Tensor:
  def __init__(self, data):
    self.data = self._convert_to_variable(data)
    
  @staticmethod
  def zeros(shape):
    def recursive_zeros(shape):
      if len(shape) == 1:
        return [Variable(0.) for _ in range(shape[0])]
      else:
        return [recursive_zeros(shape[1:]) for _ in range(shape[0])]

    return Tensor(recursive_zeros(shape))

  def _convert_to_variable(self, item):
    if isinstance(item, list):
      return [self._convert_to_variable(sub_item) for sub_item in item]
    elif not isinstance(item, Variable):
      return Variable(item)
    return item
  
  def backward(self):
    """Performs the backward pass for all Variables nested within the tensor"""
    def _backward(item):
      if isinstance(item, list):
        for sub_item in item:
          _backward(sub_item)
      elif isinstance(item, Variable):
        item.backward()
    _backward(self.data)
    
  def get_shape(self):
    dimensions = []
    
    def get_dim(sub_list):
      if isinstance(sub_list, list):
        dimensions.append(len(sub_list))
        get_dim(sub_list[0])
        
    get_dim(self.data)
    return dimensions
    
  def _apply_operation(self, x, y, op):
    if isinstance(x, list) and (y is None or isinstance(y, list)):
      if isinstance(y, list):
        return [self._apply_operation(sub_x, sub_y, op) for sub_x, sub_y in zip(x, y)]
      else:
        return [self._apply_operation(sub_x, None, op) for sub_x in x]
    else:
      return op(x, y)

  # Element-wise operations
  def _elementwise_operation(self, other, operation):
    if isinstance(other, Tensor):
      return Tensor(self._apply_operation(self.data, other.data, operation))
    else:
      return Tensor([self._apply_operation(x, other, operation) for x in self.data])

  def relu(self, other=None):
    return self._elementwise_operation(None, Variable.relu)
  
  def __add__(self, other):
    return self._elementwise_operation(other, Variable.__add__)

  def __sub__(self, other):
    return self._elementwise_operation(other, Variable.__sub__)

  def __mul__(self, other):
    return self._elementwise_operation(other, Variable.__mul__)

  def __truediv__(self, other):
    return self._elementwise_operation(other, Variable.__truediv__)

  def __neg__(self):
    return Tensor([-x for x in self.data])
  
  def __repr__(self):
    return f"Tensor(data={self.data})"
  
  # List operations
  def __getitem__(self, index):
    return Tensor(self.data[index])

  def __iter__(self):
    for sub in self.data:
      if isinstance(sub, list):
        yield Tensor(sub)
      else:
        yield sub
  
  def matmul2d(self, other):
    if not isinstance(other, Tensor):
      raise ValueError("Matrix multiplication requires another Tensor")

    self_shape = self.get_shape()
    other_shape = other.get_shape()
    if self_shape[-1] != other_shape[-2]:
      raise ValueError("Incompatible dimensions for matrix multiplication")

    # Prepare arrays for multiplication
    self_flat = self.flatten()
    other_flat = other.flatten()
    self_array_flat = np.array([v.data for v in self_flat])
    other_array_flat = np.array([v.data for v in other_flat])
    self_array = self_array_flat.reshape(self_shape)
    other_array = other_array_flat.reshape(other_shape)

    # Compute result using numpy for the forward pass
    result_array = np.matmul(self_array, other_array)

    def build_variable(i, j):
      # Track which variables were used to create each output in the result array
      row_indices = range(i * self_shape[-1], (i + 1) * self_shape[-1])
      col_indices = range(j, len(other_flat), other_shape[-1])
      children = [self_flat[r] for r in row_indices] + [other_flat[c] for c in col_indices]
      
      # Convert to Variable
      var = Variable(result_array[i, j], _children=children)
      
      # Define backward for the variable, based on matrix multiplication
      def _backward():
        grad_output = var.grad
        for k in range(self_shape[-1]):
          children[k].grad += grad_output * other_array[k, j]
        for k in range(other_shape[-2]):
          children[self_shape[-1] + k].grad += grad_output * self_array[i, k]

      var._backward = _backward
      return var

    result_tensor = Tensor([[build_variable(i, j) for j in range(result_array.shape[1])]
                            for i in range(result_array.shape[0])])
    return result_tensor
  
  # Reshaping operations
  def flatten(self):
    """Returns a flattened 1-d list of Variables"""
    def _flatten(data):
      if isinstance(data, list):
        return [x for i in data for x in _flatten(i)]
      else:
        return [data]
    return _flatten(self.data)

  def reshape(self, shape):
    """Returns a new Tensor reshaped to the given shape, where shape is a list or tuple of dimensions"""
    flat_data = self.flatten()
    it = iter(flat_data)
    
    def shape_helper(shape):
      if len(shape) == 1:
        return [next(it) for _ in range(shape[0])]
      return [shape_helper(shape[1:]) for _ in range(shape[0])]

    reshaped_data = shape_helper(shape)
    return Tensor(reshaped_data)
  
  def transpose(self):
    assert len(self.get_shape()) == 2, "Transpose can only be used for 2-d Tensors"
    return Tensor([list(row) for row in zip(*self.data)])
