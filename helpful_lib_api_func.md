# Useful Libraries
Use "!pip install <library_name>" or "!pip install <library_name> == <version_no>" if not found

# General Use (for ML)
```python
import numpy as np 
#great in matrix operations

import pandas as pd 
#great for loading data and visualizing but cumbersome at heavy operations. 
#Preferred not to use for operation. Convert to numpy array before heavy operations.

import scipy as sp
#operates on numpy useful for different types of scientific applications.

import torch 
#PyTorch Framework

import fastai
#An intuitive layered API built on PyTorch

import ml_collections
#Library of Python Collections designed for ML use cases.

import transformers
#Huggingface Transformers [NLP]

import datasets
#Huggingface Dataset and Evaluation Metrics [NLP]

import ohmeow-blurr
#integrates huggingface transformers with fastai2 [NLP]

import vision_transformer_pytorch
#Visual Transformer stuctures in PyTorch [Vision]

import wandb
#Weights and Biases for Tracking Experiments

import time
#Great to print out elapsed time

import geopandas
#Great to work with geospatial data in python
```

# Python Debugger 

```python
import pdb; 
pdb.set_trace()
```

# EDA (Exploratory Data Analysis)

```python
import matplotlib.pyplot as plt
# Most commonly used library for any type of plots

import seaborn as sns
# Great to build visually intuitive plot
```



# Visualization

```python
import pprint
#pretty print
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5] #can be tuned
```
```python
from IPython.display import Image, YouTubeVideo
Image(png_filename, width=256, height=256) # Height and Width can be tuned
YouTubeVideo('G5JT16flZwM')
```
```python
# PIL stands for Python Imaging Library.
from PIL import Image
Image.open('give image path')
```
```python
# HTML Progress Bar
from IPython.display import HTML, display
def progress(loss,value, max=100):
 return HTML(""" Batch loss :{loss}
      <progress    
value='{value}'max='{max}',style='width: 100%'>{value}
      </progress>
             """.format(loss=loss,value=value, max=max))
```
```python
# visualizing sequence of computation in pytorch autograd
import torchviz
from torchviz import make_dot
```

# NLTK (for corpus) and Gensim (word similarity)

```python
import nltk
nltk.download('reuters')
from nltk.corpus import reuters
```

```python
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
```

# Handling Randomization

```python
import random
np.random.seed(0)
random.seed(0)
```

# Decompositions

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
```

# Useful functions of the libraries

**NumPy**

```python
np.unique() 
# finds the distinct elements of a numpy array and return a flattened sorted list by default (additionally we can get the frequency counts also)
np.multiply() # element-wise multiplication
np.dot() # matrix multiplication
```

**Time**

```python
import time
start = time.time()

time.sleep(10)

done = time.time()
elapsed = done - start
print(elapsed)
```

