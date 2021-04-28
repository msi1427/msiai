# Tensor Manipulations 

```python
#convert PIL image to pixel values using numpy (np) array or PyTorch tensor 
img = Image.open('put image path here')
np.array(img)
torch.tensor(img) # can be used in GPU computations
```

```python
# Other tensor manipulations
torch.stack('list name') # convert the list of tensors into high dimensional tensors
torch_tensor_name.ndim # gets the rank
torch_tensor_name.shape # size of each axes
torch_tensor_name.type() # type of the tensor
torch_tensor_name.mean() # gets mean
torch_tensor_name.abs() # gets absolute
torch_tensor_name.sqrt() # gets square root
torch_tensor_name[start:end] # end being excluded
torch_tensor_name[:,x] # every index values of x'th column
torch_tensor_name[x.:] # every index values of x'th row
```

# torch.nn.functional

```python
import torch.nn.functional as F
F.mse_loss(torch_tensor_name) # MSE loss
F.l1_loss(torch_tensor_name) # L1 loss
```

# Automatic Gradient Calculation

```python
x = tensor(1.0).requires_grad_()
x.backward() # automatically calculate the gradients
x.grad # gradient
```

# Einsum

```python
# Best alternative of matrix Multiplication
'''
queries shape: (N, query_len, heads, heads_dim)
keys shape: (N, key_len, heads, heads_dim)
energy shape: (N, heads, query_len, key_len)
'''
energy = torch.einsum("nqhd,nkhd -> nhqk", [queries,keys]) 
# Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention. For more visit documentation
```

# unsqueeze & tril

```python
# unsqueeze => Returns a new tensor with a dimension of size one inserted at the specified position
x = torch.tensor([1, 2, 3, 4])
torch.unsqueeze(x, 0)
#tensor([[ 1,  2,  3,  4]])
torch.unsqueeze(x, 1)
#tensor([[ 1],
#        [ 2],
#        [ 3],
#        [ 4]])
```

```python
# Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
a = torch.randn(3, 3)
torch.tril(a)
```

