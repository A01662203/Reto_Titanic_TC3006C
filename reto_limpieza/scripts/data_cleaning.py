def missing_data(df_train, df_test):
    # Get missing percentage per column in df_train
    missing_values = df_train.isnull().mean() * 100
    print(missing_values)
    print('*'*50)

    # Get missing percentage per column in df_test
    missing_values_test = df_test.isnull().mean() * 100
    print(missing_values_test)

# Erase columns PassengerId and Cabin
def drop_columns(df_train, df_test):
    df_train = df_train.drop(columns=['PassengerId', 'Cabin'])
    df_test = df_test.drop(columns=['PassengerId', 'Cabin'])
    return df_train, df_test

# Erase rows from Embarked of df_train and Fare of df_test with missing values, and print me the amount of rows erased in each case
def drop_rows(df_train, df_test):
    train_rows = len(df_train)
    test_rows = len(df_test)
    df_train = df_train.dropna(subset=['Embarked'])
    df_test = df_test.dropna(subset=['Fare'])

    print(f"Instancias eliminadas en df_train: {train_rows - len(df_train)}")
    print(f"Instancias eliminadas en df_test: {test_rows - len(df_test)}")

    return df_train, df_test

def data_cleaning(df_train, df_test):
    # survival_analysis()

    missing_data(df_train, df_test)

    df_train, df_test = drop_columns(df_train, df_test)

    df_train, df_test = drop_rows(df_train, df_test)

    return df_train, df_test