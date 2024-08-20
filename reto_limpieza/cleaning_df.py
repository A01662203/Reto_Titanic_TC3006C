from reto_limpieza.scripts.data_cleaning import data_cleaning
from reto_limpieza.scripts.data_loading import data_loading
from reto_limpieza.scripts.data_transformation import data_transformation
from reto_limpieza.scripts.data_exploration import data_exploration

def cleaning_df():
    # Load the dataframes
    df_train, df_test = data_loading()
    
    # Clean the dataframes 
    df_train, df_test = data_cleaning(df_train, df_test)

    # Transform the dataframes
    df_train, df_test = data_transformation(df_train, df_test)

    # Explore the dataframes
    data_exploration(df_train, df_test)

    return df_train, df_test