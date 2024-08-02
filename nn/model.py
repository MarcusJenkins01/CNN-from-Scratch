from typing import Any


class Model:
  def __init__(self):
    pass
  
  def forward(self, x):
    pass
  
  def __call__(self, x):
    return self.forward(x)
  
  def get_parameters(self):
    return []
  
  def zero_grad(self):
    for param in self.get_parameters():
      param.grad = 0.
  