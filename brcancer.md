```python
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
```


```python
# Load dataset 
data = load_breast_cancer()
```


```python
# Organize our data 
label_names = data['target_names'] 
labels = data['target'] 
feature_names = data['feature_names'] 
features = data['data']
```


```python
# Look at our data 
print(label_names) 
print('Class label = ',labels[0]) 
print(feature_names) 
print(features[0])
```

    ['malignant' 'benign']
    Class label =  0
    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness' 'mean compactness' 'mean concavity'
     'mean concave points' 'mean symmetry' 'mean fractal dimension'
     'radius error' 'texture error' 'perimeter error' 'area error'
     'smoothness error' 'compactness error' 'concavity error'
     'concave points error' 'symmetry error' 'fractal dimension error'
     'worst radius' 'worst texture' 'worst perimeter' 'worst area'
     'worst smoothness' 'worst compactness' 'worst concavity'
     'worst concave points' 'worst symmetry' 'worst fractal dimension']
    [1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01
     1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02
     6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01
     1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01
     4.601e-01 1.189e-01]
    


```python
# Split our data 
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

```


```python
# Initialize our classifier 
gnb = GaussianNB()
```


```python
# Train our classifier 
model = gnb.fit(train, train_labels)
```


```python
# Make predictions
preds = gnb.predict(test) 
print(preds)
```


```python
# Evaluate accuracy 
print(accuracy_score(test_labels, preds))
```

    0.9414893617021277
    


```python

```
