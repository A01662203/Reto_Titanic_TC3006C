def missing_data(df_train, df_test):
    # Calcular el porcentaje de valores faltantes por columna en datos de entrenamiento
    missing_values = df_train.isnull().mean() * 100
    print(missing_values)
    print('*'*50)

    # Calcular el porcentaje de valores faltantes por columna en datos de prueba
    missing_values_test = df_test.isnull().mean() * 100
    print(missing_values_test)

# Eliminar las columnas 'PassengerId' y 'Cabin' de ambos conjuntos de datos
def drop_columns(df_train, df_test):
    df_train = df_train.drop(columns=['PassengerId', 'Cabin'])
    df_test = df_test.drop(columns=['PassengerId', 'Cabin'])
    return df_train, df_test

# Eliminar las filas con valores faltantes en las columnas 'Embarked' y 'Fare'
def drop_rows(df_train, df_test):
    train_rows = len(df_train)
    test_rows = len(df_test)
    df_train = df_train.dropna(subset=['Embarked'])
    df_test = df_test.dropna(subset=['Fare'])

    print(f"Instancias eliminadas en df_train: {train_rows - len(df_train)}")
    print(f"Instancias eliminadas en df_test: {test_rows - len(df_test)}")

    return df_train, df_test

# Limpieza de datos en ambos conjuntos de datos basada en las funciones anteriores
def data_cleaning(df_train, df_test):
    missing_data(df_train, df_test)

    df_train, df_test = drop_columns(df_train, df_test)

    df_train, df_test = drop_rows(df_train, df_test)

    return df_train, df_test