
class Dataset:
  def __init__(self, images, labels):
    self.images = images
    self.labels = labels
  
  def __iter__(self):
    return zip(self.images, self.labels)
  
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    return list(zip(self.images, self.labels))[index]
    

class Dataloader:
  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size
    self.index = 0
  
  def __iter__(self):
    self.index = 0
    return self

  def __next__(self):
    if self.index >= len(self.dataset):
      raise StopIteration
    
    batch = self.dataset[self.index:min(self.index + self.batch_size, len(self.dataset))]
    self.index += self.batch_size
    images, labels = zip(*batch)
    return list(images), list(labels)
  
  def __len__(self):
    return (len(self.dataset) + self.batch_size - 1) // self.batch_size
