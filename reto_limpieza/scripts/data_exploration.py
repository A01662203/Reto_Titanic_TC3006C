import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def survival_analysis(df_train):
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

def plt_group_size(df_train):
    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Plot a histogram for the 'Survived' column grouped by 'Group_Size', order by alone, small, medium, large, give space between bars
    sns.countplot(data=df_train, x='Group_Size', hue='Survived', order=['ALONE', 'SMALL', 'MEDIUM', 'LARGE'], dodge=True)


    # Set the title and labels
    plt.title('Survival Distribution by Family Group Size')
    plt.xlabel('Family Group Size')
    plt.ylabel('Count')
    plt.legend(['Not Survived', 'Survived'])

    # Show the plot
    plt.show()

def plt_survival_rate(df_train):
    # Make a subplot of every column in df_train as index for the pivot table ordered by the survival rate
    df_train_analisis = df_train.drop(columns=['Name', 'Age', 'Survived', 'Fare', 'TicketPrefix', 'Ticket_FirstDigit', 'Ticket_Group', 'AgeGroup', 'Group_Size', 'Ticket', 'Title', 'Ticket_Number', 'Ticket_Length'])
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Survival Rate characteristics of Male passengers')

    for i, col in enumerate(df_train_analisis.columns):
        # Create pivot table with counts for each survival status
        pivot = df_train.pivot_table(index=col, columns='Survived', aggfunc='size', fill_value=0)
        # Calculate the survival rate
        pivot['Survival Rate'] = pivot[1] / (pivot[0] + pivot[1])
        # Order the pivot table by the Survival Rate in ascending order
        pivot = pivot.sort_values(by='Survival Rate', ascending=True)
        # Plot the Survival Rate
        pivot['Survival Rate'].plot(kind='bar', ax=axs[i//3, i%3], color='skyblue')
        axs[i//3, i%3].set_title(f'Survival Rate by {col}')
        axs[i//3, i%3].set_ylabel('Survival Rate')
        axs[i//3, i%3].set_ylim(0, 1)
        axs[i//3, i%3].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def data_exploration(df_train, df_test):
    survival_analysis(df_train)

    plt_group_size(df_train)

    plt_survival_rate(df_train)