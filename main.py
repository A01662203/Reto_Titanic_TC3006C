from reto_limpieza.cleaning_df import cleaning_df

def main():
    # PARTE 1: Limpieza de los datos
    df_train, df_test = cleaning_df()

    # Generar un archivo csv con los datos limpios
    df_train.to_csv('./data/train_clean.csv', index=False)
    df_test.to_csv('./data/test_clean.csv', index=False)

    #PARTE 2: Modelado de los datos
    #modeling()

if __name__ == "__main__":
    main()