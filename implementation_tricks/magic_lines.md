## use them at the beginning of notebook (Jeremy's suggestion)

```python
# %matplotlib inline:: Ensures that all matplotlib plots will be plotted in the output cell within the notebook and will be kept in the notebook when saved.

# bs: batch size

%matplotlib inline
%reload_ext autoreload
%autoreload 2
bs = 8 
```



# notebook tricks

```python
%timeit [i+1 for i in range(1000)] # Runs a line ten thousand times and displays the average time it took to run.
%debug # Inspects a function which is showing an error using the Python debugger. If you type this in a cell just after an error, you will be directed to a console where you can inspect the values of all the variables.
type(anything) # return the type of 'anything'
```



# seed all

```python
def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_value)

seed = 777 # try with other prime numbers
seed_all(seed)
```

