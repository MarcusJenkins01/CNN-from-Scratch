class Variable:
  def __init__(self, data, _children=[]):
    self.data = data
    self.grad = 0
    self._backward = lambda: None
    self._descendents = set(_children)

  def __mul__(self, other):
    other = other if isinstance(other, Variable) else Variable(other)
    out = Variable(self.data * other.data, [self, other])
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    
    return out
  
  def __add__(self, other):
    other = other if isinstance(other, Variable) else Variable(other)
    out = Variable(self.data + other.data, [self, other])
    
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    
    return out
  
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
        
  def backward(self):
    graph = []
    visited_nodes = set()
    
    def build_graph(node):
      if node not in visited_nodes:
        visited_nodes.add(node)
        for child in node._descendents:
          build_graph(child)
        graph.append(node)
    build_graph(self)
    
    self.grad = 1
    for node in reversed(graph):
      node._backward()
  
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
