# naive-bayes-algorithm-from-scratch
This repository contains a Python implementation of the Naive Bayes algorithm built from scratch. It performs classification tasks, predicts probabilities, generates confusion matrices, and Provides a detailed classification report including precision, recall, and F1-score without relying on additional libraries.

## Example Usecase
```python
from classification import naive_bayes
model=naive_bayes()

# fit the training data to the model
model.fit(x_train,y_train)

# pass the test data to predict the target feature
model.predict(x_test)

# pass the test data to calculate the accuracy
model.score(y_test)

# access the actual and predicted values using actual_pred_table atrribute
model.actual_pred_table

# confusion matrix
model.confusion_matrix

# classification report includes precision,recall and f1-score
model.classification_report
