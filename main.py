from reto_limpieza.cleaning_df import cleaning_df
from reto_limpieza.scripts.data_cleaning import missing_data

def check_agegroup_sex_zero(df_train, df_test):
    agegroup_sex_columns = [f'AgeGroup_Sex_{i}' for i in range(29)]  # columnas de AgeGroup_Sex del 0 al 28

    # Verificar en df_train
    all_zeros_train = df_train[agegroup_sex_columns].sum(axis=1) == 0
    if all_zeros_train.any():
        print("Warning: Hay filas en df_train donde todas las columnas de AgeGroup_Sex son 0.")
        print(df_train[all_zeros_train])

    # Verificar en df_test
    all_zeros_test = df_test[agegroup_sex_columns].sum(axis=1) == 0
    if all_zeros_test.any():
        print("Warning: Hay filas en df_test donde todas las columnas de AgeGroup_Sex son 0.")
        print(df_test[all_zeros_test])

def main():
    # PARTE 1: Limpieza de los datos
    df_train, df_test = cleaning_df()

    check_agegroup_sex_zero(df_train, df_test)

    #missing_data(df_train, df_test)

    # Generar un archivo csv con los datos limpios
    df_train.to_csv('./data/train_clean.csv', index=False)
    df_test.to_csv('./data/test_clean.csv', index=False)

    #PARTE 2: Modelado de los datos
    #modeling()

if __name__ == "__main__":
    main()