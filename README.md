<br />
<div align="center">
  <h3 align="center">CNN from Scratch</h3>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project
This project implements a convolutional neural network (CNN) for image classification from scratch, using only pure Python. This demonstrates how CNNs work on a low level for educational purposes.
    
Features:
* The Variable class for tracking of gradients and back-propagation.
* The Tensor class for multi-dimensional tensors, consisting of Variable objects.
* Element-wise and matrix operations for tensors, including ReLU activation function.
* The ConvolutionalLayer2d which implements a convolutional layer using the Tensor operations.

## Implementation Detail
The ConvolutionalLayer2d uses the GEMM method to obtain the dot product of each filter for each patch as a single matrix operation.
<br>
<br>
<img src="https://github.com/user-attachments/assets/5ba05129-5ab2-4c8f-8f66-9c46d6b0757a" width="600">
<br>
<img src="https://github.com/user-attachments/assets/ec61355a-4b42-4ca1-a4c8-034210246d62" width="600">

<p align="right">(<a href="#readme-top">back to top</a>)</p>
