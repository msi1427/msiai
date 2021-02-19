# Basics 

```python
doc('put function name here') # brings out the fastai documentation of that function
'function'? # brings out the fastai link to the source code of that function
'function'?? # brings out the fastai source code of that function
show_image('put the image here') # show tensor images
plot_function(function_name,'x-axis-label','y-axis-label')
```

# Learner

```python
learn.fit() # when training from scratch
learn.fine_tune() # when training with transfer learning
learn.predict() # predict mode
```

# fastcore 

```python
# new addition to fastai
untar_data('put url here') # downloads and unzips the files if not done and returns the path
Path.BASE_PATH = 'put path here' # sets this as the base path and every path query include that by default
path.ls() # shows what else is in the path
```

