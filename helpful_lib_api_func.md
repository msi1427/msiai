# Useful Libraries
Use "!pip install <library_name>" or "!pip install <library_name> == <version_no>" if not found

# General Use
```python
import numpy as np 
#great in matrix operations
import pandas as pd 
#great for loading data but very cumbersome and problematic at operations. 
#Preferred not to use for operation. Convert to numpy array before operations.
import scipy as sp
#operates on numpy useful for different types of scientific applications.
```


# Visualization

```python
import pprint
#pretty print
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5] #can be tuned
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
```python
# NUMPY LIB
np.unique() 
# finds the distinct elements of a numpy array and return a flattened sorted list by default (additionally we can get the frequency counts also)
np.multiply() # element-wise multiplication
np.dot() # matrix multiplication
```  