import pandas as pd

def load_data():
    # Open test.csv and count the number of rows
    df_test = pd.read_csv('./data/test.csv')
    # print(df_test.shape[0])

    # Open train.csv and count the number of rows
    df_train = pd.read_csv('./data/train.csv')
    # print(df_train.shape[0])

    # Return the two dataframes
    return df_train, df_test