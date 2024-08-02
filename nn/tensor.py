from .variable import Variable


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
    if isinstance(item, Tensor):
      return item.data
    elif isinstance(item, list):
      return [self._convert_to_variable(sub_item) for sub_item in item]
    elif not isinstance(item, Variable):
      return Variable(item)
    return item
  
  def backward(self):
    """Performs the backward pass for all Variables nested within the tensor"""
    flat_data = self.flatten()
    for var in flat_data:
      var.backward()
    
  def get_shape(self):
    dimensions = []
    
    def get_dim(sub_list):
      if isinstance(sub_list, list):
        dimensions.append(len(sub_list))
        get_dim(sub_list[0])
        
    get_dim(self.data)
    return dimensions
    
  def _apply_operation(self, x, y, op):
    if isinstance(x, list):
      if y is None:
        return [self._apply_operation(sub_x, None, op) for sub_x in x]
      else:
        return [self._apply_operation(sub_x, sub_y, op) for sub_x, sub_y in zip(x, y)]
    else:
      if y is None:
        return op(x)
      else:
        return op(x, y)

  # Element-wise operations
  def _elementwise_operation(self, other, operation):
    original_shape = self.get_shape()
    
    if isinstance(other, Tensor):
      result_data = self._apply_operation(self.data, other.data, operation)
    elif other is None:
      result_data = self._apply_operation(self.data, None, operation)
    else:
      result_data = [self._apply_operation(x, other, operation) for x in self.data]

    result_tensor = Tensor(result_data)
    return result_tensor.reshape(original_shape)

  def relu(self, other=None):
    return self._elementwise_operation(None, Variable.relu)
  
  def softmax(self):
    original_shape = self.get_shape()
    data_flat = self.flatten()
    
    max_var = max(data_flat, key=lambda x: x.data)
    exps = [(v - max_var).exp() for v in data_flat]
    sum_of_exps = sum(exps)
    
    softmax_output = [x / sum_of_exps for x in exps]
    return Tensor(softmax_output).reshape(original_shape)
  
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
    sub = self.data[index]
    if isinstance(sub, list):
      return Tensor(sub)
    else:
      return sub

  def __iter__(self):
    for sub in self.data:
      yield Tensor(sub)
  
  def __len__(self):
    return len(self.data)
  
  # Matrix multiplication
  def matmul2d(self, other):
    if not isinstance(other, Tensor):
      raise ValueError("Matrix multiplication requires another Tensor")

    self_shape = self.get_shape()
    other_shape = other.get_shape()
    if self_shape[-1] != other_shape[0]:
      raise ValueError("Incompatible dimensions for matrix multiplication")

    result = []
    for i in range(self_shape[0]):
      result_row = []
      
      for j in range(other_shape[1]):
        sum_product = Variable(0.)
        
        for k in range(self_shape[-1]):
          sum_product = sum_product + (self.data[i][k] * other.data[k][j])
        result_row.append(sum_product)
      result.append(result_row)

    return Tensor(result)
  
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
