import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score
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

# Use GridSearchCV to find best parameters
param_grid = {
    'C': [0.001, 0.01, 0.1, 1., 10.],
    'penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(LogisticRegression(class_weight='balanced', solver='liblinear'), param_grid, cv=5, scoring='precision')
grid_search.fit(x_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')

# Predict with best parameters
y_pred = grid_search.predict(x_test)
precision = grid_search.score(x_test, y_test)
print(f'Precision: {precision.round(2)*100}%')


# Confusion Matrix plot
confusion_matrix_2 = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(confusion_matrix_2, annot=True, fmt='d')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.show()