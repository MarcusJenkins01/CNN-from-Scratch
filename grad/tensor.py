from variable import Variable


class Tensor:
  def __init__(self, data):
      self.data = self._convert_to_variable(data)

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

  def relu(self, other):
    return self._elementwise_operation(other, Variable.relu)
  
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
  
  # Dot product
  def dot(other):
    pass
  
  # Reshaping operations
  def flatten(self, data):
    """Returns a flattened 1-d Tensor"""
    if isinstance(data, list):
      return [a for i in data for a in self.flatten(i)]
    else:
      return [data]

  def reshape(self, shape):
    """Returns a new Tensor reshaped to the given shape, where shape is a list or tuple of dimensions"""
    flat_data = self.flatten(self.data)
    it = iter(flat_data)
    
    def shape_helper(shape):
      if len(shape) == 1:
        return [next(it) for _ in range(shape[0])]
      return [shape_helper(shape[1:]) for _ in range(shape[0])]

    reshaped_data = shape_helper(shape)
    return Tensor(reshaped_data)


if __name__ == "__main__":
  tensor1 = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  tensor2 = Tensor([[5, 2, 8], [2, 4, 6], [6, 4, 2]])
  tensor3 = tensor1 * tensor2
  tensor3.backward()
  print(tensor1)
