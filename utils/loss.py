from nn import Variable

def cross_entropy_loss(outputs, labels):
  """Calculate the cross-entropy loss given the softmax output and labels"""
  num_classes = labels.get_shape()[-1]
  batch_size = len(labels)

  loss = Variable(0.)
  for bi in range(batch_size):
    for ci in range(num_classes):
      if labels.data[bi][ci].data == 1.0:
        loss -= (labels.data[bi][ci] * outputs.data[bi][ci].log())
          
  return loss / batch_size
