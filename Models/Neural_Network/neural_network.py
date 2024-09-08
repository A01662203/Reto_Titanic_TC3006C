from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.metrics import Precision
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import matplotlib.pyplot as plt
import numpy as np

def create_model(input_shape, dropout_rate=0.0, learning_rate=0.001, optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    if dropout_rate > 0.0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[Precision()])
    return model

def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    param_grid = {
        'dropout_rate': [0.0, 0.2],
        'learning_rate': [0.001, 0.01],
        'optimizer': ['adam', 'rmsprop'],
        'batch_size': [20, 50],
        'epochs': [50, 100]
    }
    
    best_score = -np.inf
    best_params = {}

    # Definimos EarlyStopping para que detenga el entrenamiento si no mejora la métrica 'val_precision'
    early_stopping = EarlyStopping(monitor='val_precision', patience=5, restore_best_weights=True)

    for dropout_rate in param_grid['dropout_rate']:
        for learning_rate in param_grid['learning_rate']:
            for optimizer in param_grid['optimizer']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        model = create_model(X_train.shape[1:], dropout_rate, learning_rate, optimizer)
                        
                        # Entrenamos el modelo con EarlyStopping y validación en un 20% del training set
                        model.fit(
                            X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_split=0.2,  # Usamos una parte de X_train para validación
                            callbacks=[early_stopping],
                            verbose=0
                        )

                        score = model.evaluate(X_test, y_test, verbose=0)[1]  # Evaluamos la métrica de precision en el set de test

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'optimizer': optimizer,
                                'batch_size': batch_size,
                                'epochs': epochs
                            }

    return best_score, best_params

def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    plt.figure(figsize=(8, 6))
    plt.matshow(conf_matrix, cmap='Blues', fignum=1)
    plt.title('Matriz de Confusión de Red Neuronal', pad=20, fontsize=18)
    plt.xlabel('Predicción', fontsize=14)
    plt.ylabel('Real', fontsize=14)

    for (i, j), val in np.ndenumerate(conf_matrix):
        plt.text(j, i, f'{val}', ha='center', va='center', fontsize=16, color="black")

    plt.gca().set_frame_on(False)

    plt.savefig('./Models/Neural_Network/confusion_matrix_nn.png')
    plt.show()

def neural_network(X_train, y_train, X_test, y_test):
    best_score, best_params = hyperparameter_tuning(X_train, y_train, X_test, y_test)

    print(f"Mejor accuracy: {best_score}")
    print(f"Mejores hiperparámetros: {best_params}")

    best_model = create_model(X_train.shape[1:], best_params['dropout_rate'], best_params['learning_rate'], best_params['optimizer'])

    best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

    y_pred_probs = best_model.predict(X_test)
    y_pred = np.where(y_pred_probs > 0.5, 1, 0)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy en el conjunto de validación: {accuracy}")

    precision = precision_score(y_test, y_pred)
    print(f'Precision: {precision:.8f}')

    plot_confusion_matrix(y_test, y_pred)