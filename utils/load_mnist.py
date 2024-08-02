import struct
from array import array
import os


def load_mnist(dataset_dir, dataset="train"):
  num_classes = 10
  assert dataset in ["train", "test"], "Dataset must be 'train' or 'test'"
  
  if dataset == "train":
    images_path = os.path.join(dataset_dir, "train-images.idx3-ubyte")
    labels_path = os.path.join(dataset_dir, "train-labels.idx1-ubyte")
  elif dataset == "test":
    images_path = os.path.join(dataset_dir, "t10k-images.idx3-ubyte")
    labels_path = os.path.join(dataset_dir, "t10k-labels.idx1-ubyte")

  with open(images_path, "rb") as f_images:
    _, n_images, rows, cols = struct.unpack(">IIII", f_images.read(16))
    image_data = array("B", f_images.read())
    f_images.close()
  
  with open(labels_path, "rb") as f_labels:
    _, n_labels = struct.unpack(">II", f_labels.read(8))
    label_data = array("b", f_labels.read())
    f_labels.close()
  
  # Convert from 1d byte array to n_images x height x width
  images = []
  for n in range(n_images):
    images.append([])
    for i in range(rows):
      images[n].append([x / 255. for x in image_data[(n * rows * cols + i * cols):(n * rows * cols + (i + 1)* cols)]])
  
  # Convert labels to one-hot encoding
  labels = []
  for n in range(n_labels):
    labels.append([])
    labels[n] = [0.] * num_classes
    labels[n][label_data[n]] = 1.
    
  return labels, images, n_images, rows, cols
