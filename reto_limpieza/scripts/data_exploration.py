import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Función para realizar un análisis de supervivencia
def survival_analysis(df_train):
    # Imprimir el total de pasajeros que sobrevivieron y no sobrevivieron
    survival_counts = df_train.groupby('Survived').size()
    print(f'Total de pasajeros que sobrevivieron: {survival_counts[1]} | Total de pasajeros que no sobrevivieron: {survival_counts[0]}')
    print('*'*40)

    # Filtrar la supervivencia de los pasajeros masculinos 
    men = df_train[df_train['Sex'] == 'male']
    men_survival_counts = men.groupby('Survived').size()
    print(f'Pasajeros hombres que sobreviveron: {men_survival_counts[1]} | Pasajeros hombres que no sobrevivieron: {men_survival_counts[0]}')
    print('*'*40)

    # Filtrar la supervivencia de los pasajeros femeninos
    female = df_train[df_train['Sex'] == 'female']
    female_survival_counts = female.groupby('Survived').size()
    print(f'Pasajeros mujeres que sobrevivieron: {female_survival_counts[1]} | Pasajeros mujeres que no sobrevivieron: {female_survival_counts[0]}')

# Función para graficar la distribución de la supervivencia por tamaño de grupo familiar
def plt_group_size(df_train):
    # Creación de la figura
    plt.figure(figsize=(10, 6))

    # Graficar la distribución de la supervivencia por tamaño de grupo familiar
    sns.countplot(data=df_train, x='Group_Size', hue='Survived', order=['ALONE', 'SMALL', 'MEDIUM', 'LARGE'], dodge=True)

    # Definición de los títulos y etiquetas de los ejes
    plt.title('Survival Distribution by Family Group Size')
    plt.xlabel('Family Group Size')
    plt.ylabel('Count')
    plt.legend(['Not Survived', 'Survived'])

    # Mostrar la gráfica
    plt.show()

# Función para graficar la tasa de supervivencia por características de los pasajeros
def plt_survival_rate(df_train):
    # Hacer una gráfica de barras para cada columna en los datos de entrenaminto como índice de la tabla pivote ordenada por la tasa de supervivencia
    df_train_analisis = df_train.drop(columns=['PassengerId', 'Name', 'Age', 'Survived', 'Fare', 'TicketPrefix', 'Ticket_FirstDigit', 'Ticket_Group', 'AgeGroup', 'Group_Size', 'Ticket', 'Title', 'Ticket_Number', 'Ticket_Length'])
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Survival Rate characteristics of Male passengers')

    # Iterar sobre cada columna en los datos de entrenamiento
    for i, col in enumerate(df_train_analisis.columns):
        # Crear una tabla pivote con la columna actual
        pivot = df_train.pivot_table(index=col, columns='Survived', aggfunc='size', fill_value=0)

        # Calcular la tasa de supervivencia
        pivot['Survival Rate'] = pivot[1] / (pivot[0] + pivot[1])

        # Ordenar la tabla pivote por la tasa de supervivencia
        pivot = pivot.sort_values(by='Survival Rate', ascending=True)

        # Graficar la tasa de supervivencia
        pivot['Survival Rate'].plot(kind='bar', ax=axs[i//3, i%3], color='skyblue')
        axs[i//3, i%3].set_title(f'Survival Rate by {col}')
        axs[i//3, i%3].set_ylabel('Survival Rate')
        axs[i//3, i%3].set_ylim(0, 1)
        axs[i//3, i%3].grid(axis='y', linestyle='--', alpha=0.6)

    # Ajustar el espacio entre las gráficas y mostrar la figura
    plt.tight_layout()
    plt.show()

# Función para eliminar las columnas que ya no son necesarias para la parte del modelado
def drop_last_cols(df_train, df_test):
    # Eliminar las columnas 'Ticket', 'AgeGroup', 'Ticket_Number', 'Ticket_Length' y 'Title' de ambos conjuntos de datos
    df_train = df_train.drop(columns=['Ticket', 'Ticket_Number', 'Ticket_Length', 'Title', 'Name', 'TicketPrefix', 'Ticket_FirstDigit', 'Ticket_Group', 'Parch', 'SibSp'])
    df_test = df_test.drop(columns=['Ticket', 'Ticket_Number', 'Ticket_Length', 'Title', 'Name', 'TicketPrefix', 'Ticket_FirstDigit', 'Ticket_Group', 'Parch', 'SibSp'])

    return df_train, df_test

def embarked_tranformation(df_train, df_test):
    ohe = OneHotEncoder()

    # Fit on df_train and transform both df_train and df_test
    X_train = ohe.fit_transform(df_train[['Embarked']]).toarray()
    X_test = ohe.transform(df_test[['Embarked']]).toarray()

    # Asignar nombres personalizados a las columnas
    column_names = ["Embarked_Cherbourg", "Embarked_Queenstown", "Embarked_Southampton"]

    # Create DataFrames for the transformed data with custom column names
    df_train_ohe = pd.DataFrame(X_train, columns=column_names)
    df_test_ohe = pd.DataFrame(X_test, columns=column_names)

    # Concatenate the one-hot encoded columns back to the original DataFrames
    df_train = pd.concat([df_train.reset_index(drop=True), df_train_ohe], axis=1).drop(columns=['Embarked'])
    df_test = pd.concat([df_test.reset_index(drop=True), df_test_ohe], axis=1).drop(columns=['Embarked'])

    return df_train, df_test

def group_size_tranformation(df_train, df_test):
    lbl = LabelEncoder()

    # df_train
    df_train['Group_Size'] = lbl.fit_transform(df_train['Group_Size'])

    # df_test
    df_test['Group_Size'] = lbl.fit_transform(df_test['Group_Size'])

    return df_train, df_test

def sex_tranformation(df_train, df_test):
    ohe = OneHotEncoder()

    # Fit on df_train and transform both df_train and df_test
    X_train = ohe.fit_transform(df_train[['Sex']]).toarray()
    X_test = ohe.transform(df_test[['Sex']]).toarray()

    # Asignar nombres personalizados a las columnas
    column_names = ['Female', 'Male']

    # Create DataFrames for the transformed data with custom column names
    df_train_ohe = pd.DataFrame(X_train, columns=column_names)
    df_test_ohe = pd.DataFrame(X_test, columns=column_names)

    # Concatenate the one-hot encoded columns back to the original DataFrames
    df_train = pd.concat([df_train.reset_index(drop=True), df_train_ohe], axis=1).drop(columns=['Sex'])
    df_test = pd.concat([df_test.reset_index(drop=True), df_test_ohe], axis=1).drop(columns=['Sex'])

    return df_train, df_test

# Función para realizar la exploración de los datos, que incluye análisis, gráficas y eliminación de columnas innecesarias para el modelado
def data_exploration(df_train, df_test):
    survival_analysis(df_train)

    # plt_group_size(df_train)

    # plt_survival_rate(df_train)

    df_train, df_test = embarked_tranformation(df_train, df_test)

    df_train, df_test = group_size_tranformation(df_train, df_test)

    df_train, df_test = sex_tranformation(df_train, df_test)

    df_train, df_test = drop_last_cols(df_train, df_test)

    return df_train, df_test