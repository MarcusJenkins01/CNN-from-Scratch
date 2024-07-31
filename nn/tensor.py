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
    if isinstance(x, list) and isinstance(y, list):
      return [self._apply_operation(sub_x, sub_y, op) for sub_x, sub_y in zip(x, y)]
    else:
      return op(x, y)

  # Element-wise operations
  def _elementwise_operation(self, other, operation):
    if isinstance(other, Tensor):
      return Tensor(self._apply_operation(self.data, other.data, operation))
    else:
      return Tensor([self._apply_operation(x, other, operation) for x in self.data])

  def relu(self):
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
  
  # Dot product
  def dot(self, other):
    if not isinstance(other, Tensor):
      raise ValueError("Dot product requires another Tensor")

    # Flatten the Tensors and extract only the data for dot product
    flat_self = self.flatten()
    flat_other = other.flatten()
    data_self = np.array([var.data for var in flat_self])
    data_other = np.array([var.data for var in flat_other])

    # Perform dot product using numpy for efficiency
    result_array = np.dot(data_self, data_other)

    # Create Variables for each result and setup the backward function
    def make_variable(result, i, j):
      var = Variable(result, flat_self + flat_other)

      def _backward():
        grad_output = var.grad
        flat_self[i].grad += grad_output * data_other[j]
        flat_other[j].grad += grad_output * data_self[i]

      var._backward = _backward
      return var

    # Resultant tensor from the dot product
    result_vars = [make_variable(result_array[i][j], i, j) for i in range(len(flat_self)) for j in range(len(flat_other))]
    result_tensor = Tensor(result_vars)

    return result_tensor
  
  def matmul(self, other):
    if not isinstance(other, Tensor):
      raise ValueError("Matrix multiplication requires another Tensor")

    # Flatten tensors to calculate matrix multiplication on the last dimensions
    self_shape = self.get_shape()
    other_shape = other.get_shape()
    if self_shape[-1] != other_shape[-2]:
      raise ValueError("Last dimension of 'self' must match second to last dimension of 'other' for matmul")

    # Prepare arrays for multiplication
    self_flat = self.flatten()
    other_flat = other.flatten()
    self_array = np.array(self_flat)
    other_array = np.array(other_flat)

    # Compute result using numpy for the forward pass
    result = np.matmul(self_array.reshape(self_shape), other_array.reshape(other_shape))
    result_tensor = Tensor([[Variable(x, self_flat + other_flat) for x in row] for row in result])

    # Setup backward function to compute gradients of the dot product
    def _backward():
      self_grad = np.matmul(result.grad, other_array.T)
      other_grad = np.matmul(self_array.T, result.grad)
      for i in range(len(self.flatten())):
        self.flatten()[i].grad += self_grad.flatten()[i]
      for i in range(len(other.flatten())):
        other.flatten()[i].grad += other_grad.flatten()[i]

    result_tensor._backward = _backward

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
