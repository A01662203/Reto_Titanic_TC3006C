from scripts.data_loading import load_data
from scripts.data_cleaning import data_cleaning, missing_data
from scripts.data_transformation import data_transformation

def main():
    # Load the dataframes
    df_train, df_test = load_data()
    
    # Clean the dataframes 
    df_train, df_test = data_cleaning(df_train, df_test)

    # Transform the dataframes
    df_train, df_test = data_transformation(df_train, df_test)

    missing_data(df_train, df_test)

if __name__ == "__main__":
    main()