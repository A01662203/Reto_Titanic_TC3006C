from reto_limpieza.cleaning_df import cleaning_df

def main():
    # PARTE 1: Limpieza de los datos
    df_train, df_test = cleaning_df()

    #PARTE 2: Modelado de los datos
    #modeling()

if __name__ == "__main__":
    main()