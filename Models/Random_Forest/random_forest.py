from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(y_test, y_pred_rf):
    # Mostrar la matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred_rf)
    print('Confusion Matrix:')
    print(conf_matrix)
    # Graficar la matriz de confusión con matshow
    plt.figure(figsize=(8, 6))
    plt.matshow(conf_matrix, cmap='Blues', fignum=1)  # Utilizar fignum=1 para evitar la creación de una nueva figura
    plt.title('Matriz de Confusión de Random Forest', pad=20, fontsize=18)
    plt.xlabel('Predicción', fontsize=14)
    plt.ylabel('Real', fontsize=14)

    # Agregar las anotaciones de los valores en la matriz
    for (i, j), val in np.ndenumerate(conf_matrix):
        plt.text(j, i, f'{val}', ha='center', va='center', fontsize=16, color="black")

    # Remover la barra de color
    plt.gca().set_frame_on(False)  # Remueve el borde alrededor de la matriz

    plt.savefig('./Models/Random_Forest/confusion_matrix_rf.png')

    plt.show()

def random_forest(X_train, y_train, X_test, y_test):
    print('Random Forest')
    rf_model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 300, 500, 700],
        'max_depth': [10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8, 16],
    }

    # Crear el modelo de GridSearchCV
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='precision', n_jobs=-1)

    print('Fitting the model...')

    # Ajustar el modelo a los datos de entrenamiento
    grid_search.fit(X_train, y_train)

    # Imprimir los mejores parámetros encontrados
    print("Best Parameters:", grid_search.best_params_)

    # Predecir los datos de prueba usando el mejor modelo encontrado por GridSearchCV
    y_pred_rf = grid_search.best_estimator_.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred_rf)
    print(f'Accuracy: {accuracy:.8f}')

    # Evaluar la precision
    precision = precision_score(y_test, y_pred_rf)
    print(f'Precision: {precision:.8f}')

    # Mostrar la matriz de confusión
    plot_confusion_matrix(y_test, y_pred_rf)
