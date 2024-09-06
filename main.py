from Cleaning.cleaning import cleaning
from Models.modeling import modeling
from decTree import decTree

def main():
    # PARTE 1: Limpieza de los datos
    # df_train, df_test = cleaning()

    # Generar un archivo csv con los datos limpios
    # df_train.to_csv('./data/train/train_clean.csv', index=False)
    # df_test.to_csv('./data/test/test_clean.csv', index=False)

    # Decision Tree
    # decTree(df_train)

    #PARTE 2: Modelado de los datos
    modeling()

if __name__ == "__main__":
    main()