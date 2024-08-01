import numpy as np


class Variable:
  def __init__(self, data, _children=(), _backward=lambda: None):
    self.data = data
    self.grad = 0.
    self._backward = _backward
    self._descendents = set(_children)
    
  def __add__(self, other):
    other = other if isinstance(other, Variable) else Variable(other)
    out = Variable(self.data + other.data, (self, other))
    
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Variable) else Variable(other)
    out = Variable(self.data * other.data, (self, other))
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    
    return out

  def backward(self):
    from collections import deque
    in_degree = {}
    for node in self._descendents:
      in_degree[node] = in_degree.get(node, 0) + 1
    
    queue = deque([v for v in self._descendents if in_degree.get(v, 0) == 0])
    result = []
    while queue:
      node = queue.popleft()
      result.append(node)
      for successor in node._descendents:
        in_degree[successor] -= 1
        if in_degree[successor] == 0:
          queue.append(successor)
    
    self.grad = np.array(1.0)
    for node in reversed(result):
      node._backward()
  
  def __pow__(self, other):
    out = Variable(self.data**other, [self])
    
    def _backward():
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    
    return out
  
  def relu(self):
    out = Variable(0 if self.data < 0 else self.data, [self])
    
    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out
  
  def __truediv__(self, other):
      return self * other**-1
  
  def __sub__(self, other):
      return self + (-other)
  
  def __neg__(self):
        return self * -1

  def __rmul__(self, other):
      return self * other
  
  def __radd__(self, other):
      return self + other

  def __rtruediv__(self, other):
      return other * self**-1
  
  def __rsub__(self, other):
      return other + (-self)

  def __repr__(self):
      return f"Variable(data={self.data}, grad={self.grad}, children={self._descendents})"
