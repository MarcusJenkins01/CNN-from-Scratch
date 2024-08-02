import math
from collections import deque


class Variable:
  def __init__(self, data, _children=(), _backward=None):
    assert not isinstance(data, Variable), "Variable.data must be a number"
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
    topo = []
    visited = set()
    stack = [self]

    while stack:
      v = stack.pop()
      if v not in visited:
        visited.add(v)
        topo.append(v)
        stack.extend(v._descendents)

    self.grad = 1.0
    for v in reversed(topo):
      if v._backward is not None:
        v._backward()
  
  def __pow__(self, other):
    other = other if isinstance(other, Variable) else Variable(other)
    out = Variable(self.data**other.data, [self, other])
    
    def _backward():
      self.grad += (other.data * self.data**(other.data - 1)) * out.grad
      other.grad += (self.data ** other.data * math.log(self.data)) * out.grad
    out._backward = _backward
    
    return out
  
  def log(self):
    if self.data <= 0:
      raise ValueError("Logarithm undefined for non-positive values")

    out = Variable(math.log(self.data), [self])

    def _backward():
      if self.data != 0:
        self.grad += out.grad / self.data
    out._backward = _backward
    
    return out
  
  def relu(self):
    out = Variable(0 if self.data < 0 else self.data, [self])
    
    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out
  
  def exp(self):
    out = Variable(math.exp(self.data), [self])

    def _backward():
      self.grad += out.grad * math.exp(self.data)
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
    return f"Variable(data={self.data}, grad={self.grad})"
