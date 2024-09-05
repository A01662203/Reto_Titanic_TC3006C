import pandas as pd

#La función data_loading se encarga de importar los conjuntos de datos y asignarlos a una variable
#Devuelve dos variables que contienen cada uno de los conjuntos de datos
def data_loading(csv_test, csv_train):
    # Abre el archivo test.csv 
    df_test = pd.read_csv(csv_test)
    # print(df_test.shape[0])

    # Abre el archivo train.csv 
    df_train = pd.read_csv(csv_train)
    # print(df_train.shape[0])

    # Devolver ambos dataframes
    return df_train, df_test