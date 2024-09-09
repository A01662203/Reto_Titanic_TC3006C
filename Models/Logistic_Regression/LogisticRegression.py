import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load train data
train = pd.read_csv('./data/train_clean.csv')
# Load test data
test = pd.read_csv('./data/test_clean.csv')

# Drop the columns that are not needed in the train dataset
x_train = train.drop('Survived', axis=1)
y_train = train['Survived']

# Drop the columns that are not needed in the test dataset
x_test = test.drop('Survived', axis=1)
y_test = test['Survived']

# Divide train dataset into train and validation datasets
X_train, X_validation, Y_train, Y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Use GridSearchCV to find best parameters
param_grid = {
    'C': [0.001, 0.01, 0.1, 1., 10.],
    'penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(LogisticRegression(class_weight='balanced', solver='liblinear'), param_grid, cv=5, scoring='precision')
grid_search.fit(X_train, Y_train)
print(f'Best parameters: {grid_search.best_params_}')

### Train dataset

# Print True Positive, False Positive, True Negative, False Negative as variables
y_pred = grid_search.predict(X_train)
confusion_matrix_1 = confusion_matrix(Y_train, y_pred)
print(f'True Positive: {confusion_matrix_1[1][1]}')
print(f'False Positive: {confusion_matrix_1[0][1]}')
print(f'True Negative: {confusion_matrix_1[0][0]}')
print(f'False Negative: {confusion_matrix_1[1][0]}')
print(f'Precision: {(precision_score(Y_train, y_pred).round(2))*100}%')
print(f'Accuracy: {(accuracy_score(Y_train, y_pred).round(2))*100}%')
print(f'Recall: {(confusion_matrix_1[1][1]/(confusion_matrix_1[1][1]+confusion_matrix_1[1][0])).round(2)*100}%')

# Delete variable y_pred
del y_pred

### Validation dataset

# Predict with best parameters
y_pred_test = grid_search.predict(X_validation)
precision = grid_search.score(X_validation, Y_validation)
print(f'Precision: {precision.round(2)*100}%')
accuracy = accuracy_score(Y_validation, y_pred_test)
print(f'Accuracy: {accuracy.round(2)*100}%')
recall = (confusion_matrix(Y_validation, y_pred_test)[1][1]/(confusion_matrix(Y_validation, y_pred_test)[1][1]+confusion_matrix(Y_validation, y_pred_test)[1][0])).round(2)*100
print(f'Recall: {recall}%')


# Confusion Matrix plot
confusion_matrix_2 = confusion_matrix(Y_validation, y_pred_test)

print(f'True Positive: {confusion_matrix_2[1][1]}')
print(f'False Positive: {confusion_matrix_2[0][1]}')
print(f'True Negative: {confusion_matrix_2[0][0]}')
print(f'False Negative: {confusion_matrix_2[1][0]}')

# Plot the confusion matrix
sns.heatmap(confusion_matrix_2, annot=True, fmt='d')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.show()