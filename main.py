from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn import svm

# loading the data
breasts= pd.read_csv('Breast_cancer_data.csv')

classes = ['beningn', 'malignant']

X = breasts.iloc[:]
y = breasts.iloc[:, -1]

# creating test and training variables with test_train_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model creation
model = svm.SVC()
# model training

model.fit(X_train, y_train)


# predictions
predictions = model.predict(X_test)
# accuracy
accuracy = accuracy_score(y_test, predictions)

print('predictions:', predictions)
print('actual:', y_test)
print('accuracy:', accuracy)


# getting the names
for i in range(len(predictions)):
    print(classes[predictions[i]])



