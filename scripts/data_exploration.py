import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = None
df_test = None

def survival_analysis():
    # Print amount of survived and not survived passengers
    # Print the amount of passengers that survived and not survived
    survival_counts = df_train.groupby('Survived').size()
    print(f'Total de pasajeros que sobrevivieron: {survival_counts[1]} | Total de pasajeros que no sobrevivieron: {survival_counts[0]}')
    print('*'*40)

    # Filter the dataset to include only men
    men = df_train[df_train['Sex'] == 'male']
    # Group by 'Survived' and count the number of men who survived and not survived
    men_survival_counts = men.groupby('Survived').size()
    print(f'Pasajeros hombres que sobreviveron: {men_survival_counts[1]} | Pasajeros hombres que no sobrevivieron: {men_survival_counts[0]}')
    print('*'*40)
    female = df_train[df_train['Sex'] == 'female']
    female_survival_counts = female.groupby('Survived').size()
    print(f'Pasajeros mujeres que sobrevivieron: {female_survival_counts[1]} | Pasajeros mujeres que no sobrevivieron: {female_survival_counts[0]}')

def data_exploration(train, test):
    global df_train, df_test
    
    df_train = train
    df_test = test

    survival_analysis()

    return df_train, df_test