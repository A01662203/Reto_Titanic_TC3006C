import pandas as pd


#La función complete_ages crea una columna llamada ¨Title¨ que toma el prefijo en base al nombre de los tripulantes. Para luego obtener el promedio de sus edades y así rellenar las edades faltantes.
#Obtiene como prarámetros el dataset de entrenamiento y el de testeo.

def complete_ages(df_train, df_test):
    # Obtener el título de la columna nombre.
    df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

    # Obtener la edad promedio para cada título extraido 
    average_age_per_title = df_train.groupby('Title')['Age'].mean()

    # Llenar los valores que hacen falta en la columna de edad utilizando el promedio de edad dependiendo de el título utilizando tomando en cuenta la desviación estandar.
    for title in average_age_per_title.index:
        df_train.loc[(df_train['Age'].isnull()) & (df_train['Title'] == title), 'Age'] = average_age_per_title[title] + df_train['Age'].std()

    # Redondear los valores correspondientes a la edad al int más cercano.
    df_train['Age'] = df_train['Age'].round()
    
    # Repetir el mismo proceso en el dataset de test.
    df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    average_age_per_title_test = df_test.groupby('Title')['Age'].mean()
    for title in average_age_per_title_test.index:
        df_test.loc[(df_test['Age'].isnull()) & (df_test['Title'] == title), 'Age'] = average_age_per_title_test[title] + df_test['Age'].std()
    df_test['Age'] = df_test['Age'].round()

    # Repetir el proceso e el dataset de testeo utilizando el promedio de edades por título del dataset de entrenamiento.
    for title in average_age_per_title.index:
        df_test.loc[(df_test['Age'].isnull()) & (df_test['Title'] == title), 'Age'] = average_age_per_title[title] + df_test['Age'].std()
    df_test['Age'] = df_test['Age'].round()

    # Eliminar la columna de ¨Title¨ de ambos conjuntos de datos.
    df_train = df_train.drop(columns=['Title'])
    df_test = df_test.drop(columns=['Title'])

    return df_train, df_test    

def group_size_col(df_train, df_test):
    Family_Count_Tr  = df_train['SibSp'] + df_train['Parch'] + 1
    Family_Count_Ts  = df_test['SibSp'] + df_test['Parch'] + 1

    def categorize_family_size(family_count):
        if family_count == 1:
            return 'ALONE'
        elif family_count in [2, 3, 4]:
            return 'SMALL'
        elif family_count in [5, 6]:
            return 'MEDIUM'
        elif family_count in [7, 8, 11]:
            return 'LARGE'
        else:
            return 'UNKNOWN'  # En caso que se encuentre un grupo no incluido anteriormente

    # Apply the function to create a new column
    Family_Size_Tr= Family_Count_Tr.apply(categorize_family_size)
    Family_Size_Ts= Family_Count_Ts.apply(categorize_family_size)

    # Add the new column to the dataframes
    df_test['Group_Size'] = Family_Size_Ts
    df_train['Group_Size'] = Family_Size_Tr

    return df_train, df_test

#La función ticket_info_col crea 3 columnas nuevas que incluyen información a partir del ticket de cada pasajero
#Recibe el conjunto de datos de prueba y de testeo
#Devuelve ambos conjuntos de datos con las columna actualizadas

def ticket_info_col(df_train, df_test):
    df_train['Ticket'] = df_train['Ticket'].replace('LINE', 'LINE 0')
    df_test['Ticket'] = df_test['Ticket'].replace('LINE', 'LINE 0')

    # Parsear la información del ticket
    df_train['Ticket'] = df_train['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())

    df_train['TicketPrefix'] = df_train['Ticket'].apply(lambda x: get_prefix(x))

    # Separar todos los componentes del ticket en el conjunto de datos de entrenamiento
    df_train['Ticket_Number'] = df_train['Ticket'].apply(lambda x: int(x.split(' ')[-1])//1)
    df_train['Ticket_Length'] = df_train['Ticket_Number'].apply(lambda x : len(str(x)))
    df_train['Ticket_FirstDigit'] = df_train['Ticket_Number'].apply(lambda x : int(str(x)[0]))
    df_train['Ticket_Group'] = df_train['Ticket'].apply(lambda x: str(int(x.split(' ')[-1])//10))

    # Separar todos los componentes del ticket en el conjunto de datos de testeo
    df_test['Ticket'] = df_test['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())
    df_test['TicketPrefix'] = df_test['Ticket'].apply(lambda x: get_prefix(x))
    df_test['Ticket_Number'] = df_test['Ticket'].apply(lambda x: int(x.split(' ')[-1])//1)
    df_test['Ticket_Length'] = df_test['Ticket_Number'].apply(lambda x : len(str(x)))
    df_test['Ticket_FirstDigit'] = df_test['Ticket_Number'].apply(lambda x : int(str(x)[0]))
    df_test['Ticket_Group'] = df_test['Ticket'].apply(lambda x: str(int(x.split(' ')[-1])//10))

    # DEliminar las columnas de Ticket, número de ticket, longitud del número de ticket 
    df_train = df_train.drop(columns=['Ticket', 'Ticket_Number', 'Ticket_Length'])
    df_test = df_test.drop(columns=['Ticket', 'Ticket_Number', 'Ticket_Length'])

    ticket_table = pd.crosstab(df_train['Pclass'],df_train['Ticket_FirstDigit'],margins=True)
    print(ticket_table)

    return df_train, df_test


# La función get_prefix extrae el prefijo del ticket
#Recibe el ticket
#Devuelve el prefijo o un string indicando que no tiene prefijo

def get_prefix(ticket):
    lead = ticket.split(' ')[0][0]
    if lead.isalpha():
        return ticket.split(' ')[0]
    else:
        return 'NoPrefix'


#La función data_transformation se encarga de llamar a otras funciones encargadas de realizar transformaciones en ambos conjuntos de datos.
#Recibe el conjunto de entrenamiento y el de testeo
#Retorna ambos conjuntos de datos actualizados

def data_transformation(df_train, df_test):
    complete_ages(df_train, df_test)

    group_size_col(df_train, df_test)

    ticket_info_col(df_train, df_test)

    return df_train, df_test

