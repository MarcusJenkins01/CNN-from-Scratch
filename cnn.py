from nn import Tensor, ConvolutionalLayer2d, Model
from utils import load_mnist, Dataset, Dataloader, cross_entropy_loss


class CNN(Model):
  def __init__(self, n_classes):
    self.n_classes = n_classes
    self.conv1 = ConvolutionalLayer2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
    self.conv2 = ConvolutionalLayer2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
    self.conv3 = ConvolutionalLayer2d(in_channels=8, out_channels=n_classes, kernel_size=28)
    # self.conv3 = ConvolutionalLayer2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
    # self.conv4 = ConvolutionalLayer2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
    # self.conv5 = ConvolutionalLayer2d(in_channels=16, out_channels=n_classes, kernel_size=28)
  
  def forward(self, x):
    batch_size, *_ = x.get_shape()
    out = self.conv1(x).relu()
    print(out.get_shape())
    out = self.conv2(out).relu()
    print(out.get_shape())
    out = self.conv3(out)
    print(out.get_shape())
    out = out.reshape((batch_size, self.n_classes))
    return out
  
  def get_parameters(self):
    all_parameters = []
    for attr in dir(self):
      attribute = getattr(self, attr)
      if isinstance(attribute, Model):
        params = attribute.get_parameters()
        all_parameters.extend(params)
        
    return all_parameters


def train(train_loader, learning_rate, num_epochs, num_classes):
  cnn = CNN(n_classes=num_classes)
  
  for epoch in range(num_epochs):
    # One epoch
    for image_batch, label_batch in iter(train_loader):
      image_batch = Tensor(image_batch).reshape((4, 1, 28, 28))
      label_batch = Tensor(label_batch).reshape((4, 10))
      out = cnn(image_batch)
      print(out.get_shape())
      
      batch_softmax = []
      for bi, batch in enumerate(out):
        batch_softmax.append(batch.softmax().flatten())
        print(batch.softmax())
        print(label_batch[bi])
        print()
      
      out = Tensor(batch_softmax)
      
      # Back-propagation
      loss = cross_entropy_loss(out, label_batch)
      cnn.zero_grad()
      loss.backward()
      
      print(loss)
      
      # Bump weights in a direction that reduces the loss
      for param in cnn.get_parameters():
        param -= param.grad * learning_rate


if __name__ == "__main__":
  learning_rate = 0.1
  num_epochs = 300
  
  labels, images, n_images, rows, cols = load_mnist(dataset_dir="dataset", dataset="train")
  train_dataset = Dataset(images=[images[:4]], labels=[labels[:4]])
  train_loader = Dataloader(dataset=train_dataset, batch_size=4)
  num_classes = 10
  
  train(train_loader, learning_rate, num_epochs, num_classes)
