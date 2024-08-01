from nn import Tensor, ConvolutionalLayer2d
from utils import load_mnist, Dataset, Dataloader


class CNN:
  def __init__(self, n_classes=10):
    self.n_classes = n_classes
    self.conv1 = ConvolutionalLayer2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv2 = ConvolutionalLayer2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv3 = ConvolutionalLayer2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv4 = ConvolutionalLayer2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv5 = ConvolutionalLayer2d(in_channels=16, out_channels=n_classes, kernel_size=28)
  
  def forward(self, x):
    batch_size, *_ = x.get_shape()
    out = self.conv1(x).relu()
    out = self.conv2(out).relu()
    out = self.conv3(out).relu()
    out = self.conv4(out).relu()
    out = self.conv5(out)
    out = out.reshape((batch_size, self.n_classes))
    return out
  
  def __call__(self, x):
    return self.forward(x)


if __name__ == "__main__":
  cnn = CNN()
  labels, images, n_images, rows, cols = load_mnist(dataset_dir="dataset", dataset="train")
  train_dataset = Dataset(images=images, labels=labels)
  train_loader = Dataloader(dataset=train_dataset, batch_size=4)
  
  for image_batch, label_batch in train_loader:
    image_batch = Tensor(image_batch).reshape((4, 1, 28, 28))
    label_batch = Tensor(label_batch).reshape((4, 1))
    out = cnn(image_batch)
    print(out.get_shape())
    out.backward()
    
    for bi, batch_out in enumerate(out.data):
      for out_bi in batch_out:
        print(out_bi)
