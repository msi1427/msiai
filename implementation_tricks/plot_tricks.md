# Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
mat = confusion_matrix(predicted, actual)
heat = sns.heatmap(mat,square=True,annot=True,fmt='d',cbar=True,cmap=plt.cm.gist_heat)
class_label=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
heat.set_xticklabels(class_label, rotation=90)
heat.set_yticklabels(class_label, rotation=0)
heat.set_xlabel('Predicted Label')
heat.set_ylabel('True Label')
```

# Barplot

```python
ax = sns.barplot(x = class_names, y = class_acc)
ax.set_xlabel('Type of Clothing',color="white")
ax.set_ylabel('Accuracy',color="white")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90,color="white") 
```



# Plot more than one in a single graph with legends

```python
plt.plot(epoch_list, train_acc,label='train_acc')
plt.plot(epoch_list, test_acc,label='test_acc')
plt.title("Accuracy comparison on each epoch",color="white")
plt.xlabel("Epochs",color="white")
plt.ylabel("Accuracy",color="white")
plt.legend(loc="upper left")
plt.show()
```