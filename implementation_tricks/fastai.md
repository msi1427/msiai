# Basics 

```python
doc('put function name here') # brings out the fastai documentation of that function
'function'? # brings out the fastai link to the source code of that function
'function'?? # brings out the fastai source code of that function
show_image('put the image here') # show tensor images
plot_function(function_name,'x-axis-label','y-axis-label')
```

# fastai in Google Colab

```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
```

# Learner

```python
learn.fit() # when training from scratch
learn.fine_tune() # when training with transfer learning
learn.predict() # predict mode

# cnn_learner is the default fastai learner for convnets
learn = cnn_learner('put dataloader here', 'put the model architecture', metrics=[error_rate,accuracy])
```

# fastcore 

```python
# new addition to fastai
untar_data('put url here') # downloads and unzips the files if not done and returns the path
Path.BASE_PATH = 'put path here' # sets this as the base path and every path query include that by default
path.ls() # shows what else is in the path
```

# Plot Confusion Matrix

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(8, nrows=2)
```

# Data Cleaning

```python
from fastai.vision.widgets import *
cleaner = ImageClassifierCleaner('put the trained model here')
cleaner
```

```python
for idx in cleaner.delete(): cleaner.fns[idx].unlink()                            # delete irrelevant data
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat/cat) # change the directory
```