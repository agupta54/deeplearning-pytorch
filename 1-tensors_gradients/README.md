## Overview of tensors in PyTorch

A neural network is a function. Py-Torch neural networks are composed of Python centers. 

It consists of an input tensor $ x $ and an output tensor $ y $ and the network will be composed of a set of parameters. These are also tensors. A neural network will be comprised of a series of tensor operations. 

Tensor can be of many types like double or integer. we can convert to a rectangular tensor. It is not so difficult to convert numpy arrays to pytorch tensors and vice versa. This gives pytorch tremendous capabilities to work in the python ecosystem. 

Parameters are a kind of tensor that will allow you to calculate the gradient or derivatives that will allow you to train the neural network. We can do this in pytorch by setting the parameter required grad equal to true.

## Tensors in One-Dimension 

```
import torch 
import numpy as np 
a = torch.tensor([0,1,2,3,4])
a.dtype 
a.type()
a = torch.FloatTensor([0,1,2,3,4]) # casting integer as float
a = a.type(torch.FloatTensor)
a.size()
a.dimension()
a_col = a.view(5,1)
a_col = a.view(-1,1) # use -1 if we don't know actual length of tensor 
np_a = np.array([0,1,2,3,4,5,6])
torch_a = torch.from_numpy(np_a)
back_np = torch_a.numpy()
```
All elements have to be the same data type inside the tensor and the indexing starts from 0. We can access each element by the usual [ ]. Slicing is same as numpy arrays as well. 

### Basic Operations  

```
# Vector addition is element wise 
u = torch.tensor([1,0])
v = torch.tensor([0,1])
z = u+v
# Scalar multiplication is element wise 
y = torch.tensor([1,2])
z = 2*y
# Product of two tensors 
u = torch.tensor([1,2])
v = torch.tensor([3,2])
z = u*v # by default this is element wise product 
result = torch.dot(u,v) # dot product 
u = torch.tensor([1,2,3,-1])
z = u + 1 # broadcasting 
a = torch.tensor([1.0, -1, 1, -1])
mean_a = a.mean()
max_a = a.max() 
x = torch.linspace(-2,2,steps=5)
x = torch.linspace(0,2*np.pi,100)
y = torch.sin(x)
import matplotlib.pylot as plt 
plt.plot(x.numpy(), y.numpy())
plt.show()
```
## Tensors in 2-Dimension

A 2-D tensor is a container of information. Each column in a tensor is called a feature and each row is a sample.
Grayscale images can be represented as 2-D tensors. 0 correponds to black and 255 corresponds to white. 
Tensors can be extended to 3 dimensions, 4 dimensions and so on. 

```
a = [[11,12,13], [21,22,23], [31,32,33]]
A = torch.tensor(a)
A.ndimension()
A.shape()
A.size()
A = torch.tensor([[0,1,2],[1,0,1]])
B = torch.tensor([[1,1],[1,1],[-1,1]])
C = torch.mm(A,B)
```
