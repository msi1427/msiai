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

