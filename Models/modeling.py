from Cleaning.scripts.data_loading import data_loading
from Models.Neural_Network.neural_network import neural_network
from Models.Random_Forest.random_forest import random_forest

def modeling():
    # Cargar los dataframes de sus respectivos archivos csv
    df_train, df_test = data_loading('./data/test_clean.csv', './data/train_clean.csv')

    y_train = df_train['Survived']
    x_train = df_train.drop(columns=['Survived'])
    y_test = df_test['Survived']
    x_test = df_test.drop(columns=['Survived'])

    # MODELO 1: Regresión Logística

    # MODELO 2: Random Forest
    random_forest(x_train, y_train, x_test, y_test)

    # MODELO 3: Neural Network
    neural_network(x_train, y_train, x_test, y_test)
    
    return 0