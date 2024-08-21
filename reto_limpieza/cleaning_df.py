from reto_limpieza.scripts.data_cleaning import data_cleaning
from reto_limpieza.scripts.data_loading import data_loading
from reto_limpieza.scripts.data_transformation import data_transformation
from reto_limpieza.scripts.data_exploration import data_exploration

def cleaning_df():
    # Cargar los dataframes de sus respectivos archivos csv
    df_train, df_test = data_loading()
    
    # Análisis de datos faltantes y eliminación de columnas y filas por dicho motivo
    df_train, df_test = data_cleaning(df_train, df_test)

    # Completado de edades faltantes, creación de columna 'Group_Size',  'AgeGroup' y 'AgeGroupSex', y separación de la columna 'Ticket'
    df_train, df_test = data_transformation(df_train, df_test)

    # Análisis de sobrevivientes y creación de gráficos
    df_train, df_test = data_exploration(df_train, df_test)

    return df_train, df_test