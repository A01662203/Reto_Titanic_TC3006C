import pandas as pd

def complete_ages(df_train, df_test):
    # Obtain the Title from the Name column
    df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

    # Obtain the average age per title
    average_age_per_title = df_train.groupby('Title')['Age'].mean()

    # Fill missing values in the Age column with the average age per title with a random number between a standard deviation of 1
    for title in average_age_per_title.index:
        df_train.loc[(df_train['Age'].isnull()) & (df_train['Title'] == title), 'Age'] = average_age_per_title[title] + df_train['Age'].std()

    # Round the age to the nearest integer
    df_train['Age'] = df_train['Age'].round()
    
    # Repeat the same process for the test dataset
    df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    average_age_per_title_test = df_test.groupby('Title')['Age'].mean()
    for title in average_age_per_title_test.index:
        df_test.loc[(df_test['Age'].isnull()) & (df_test['Title'] == title), 'Age'] = average_age_per_title_test[title] + df_test['Age'].std()
    df_test['Age'] = df_test['Age'].round()

    # Repeat the same process for the test dataset using the train average age per title
    for title in average_age_per_title.index:
        df_test.loc[(df_test['Age'].isnull()) & (df_test['Title'] == title), 'Age'] = average_age_per_title[title] + df_test['Age'].std()
    df_test['Age'] = df_test['Age'].round()

    # Drop Title columns in both datasets
    df_train = df_train.drop(columns=['Title'])
    df_test = df_test.drop(columns=['Title'])

    return df_train, df_test    

def group_size_col(df_train, df_test):
    Family_Count_Tr  = df_train['SibSp'] + df_train['Parch'] + 1
    Family_Count_Ts  = df_test['SibSp'] + df_test['Parch'] + 1

    def categorize_family_size(family_count):
        if family_count == 1:
            return 'ALONE'
        elif family_count in [2, 3, 4]:
            return 'SMALL'
        elif family_count in [5, 6]:
            return 'MEDIUM'
        elif family_count in [7, 8, 11]:
            return 'LARGE'
        else:
            return 'UNKNOWN'  # In case there are other sizes not covered

    # Apply the function to create a new column
    Family_Size_Tr= Family_Count_Tr.apply(categorize_family_size)
    Family_Size_Ts= Family_Count_Ts.apply(categorize_family_size)

    # Add the new column to the dataframes
    df_test['Group_Size'] = Family_Size_Ts
    df_train['Group_Size'] = Family_Size_Tr

    return df_train, df_test

def age_group_sex_col(df_train, df_test):
    # Segment by age in intervals of 5 of passengers and using pivot tables to analyze the survival rate
    age_bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    age_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']

    df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=age_bins, labels=age_labels)
    df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=age_bins, labels=age_labels)

    # Paso 1: Contar las ocurrencias conjuntas de Group Age y Sex
    joint_counts = df_train.groupby(['Sex', 'AgeGroup'], observed=False).size().unstack(fill_value=0)

    # Paso 2: Calcular la probabilidad condicional P(Group Age | Sex)
    conditional_probabilities = joint_counts.div(joint_counts.sum(axis=1), axis=0)
    #print(conditional_probabilities*100)

    # Generar una columna donde sea el ['AgeGroup'] y el ['Sex'] en una sola columna
    df_train['AgeGroup_Sex'] = df_train['AgeGroup'].astype(str) + '_' + df_train['Sex']
    df_test['AgeGroup_Sex'] = df_test['AgeGroup'].astype(str) + '_' + df_test['Sex']

    # Paso 1: Contar las ocurrencias conjuntas de Sex, Group Age y Survived
    joint_counts_survived = df_train.groupby(['Sex', 'AgeGroup', 'Survived'], observed=False).size().unstack(fill_value=0)

    # Paso 2: Calcular la probabilidad condicional P(Survived = 1 | Sex, Group Age)
    survived_counts = joint_counts_survived[1]
    conditional_probabilities_survived = survived_counts.div(joint_counts_survived.sum(axis=1))
    #print(conditional_probabilities_survived*100)

    return df_train, df_test

def ticket_info_col(df_train, df_test):
    df_train['Ticket'] = df_train['Ticket'].replace('LINE', 'LINE 0')
    df_test['Ticket'] = df_test['Ticket'].replace('LINE', 'LINE 0')

    # Parse Ticket feature
    df_train['Ticket'] = df_train['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())

    df_train['TicketPrefix'] = df_train['Ticket'].apply(lambda x: get_prefix(x))

    # Separate all ticket components
    df_train['Ticket_Number'] = df_train['Ticket'].apply(lambda x: int(x.split(' ')[-1])//1)
    df_train['Ticket_Length'] = df_train['Ticket_Number'].apply(lambda x : len(str(x)))
    df_train['Ticket_FirstDigit'] = df_train['Ticket_Number'].apply(lambda x : int(str(x)[0]))
    df_train['Ticket_Group'] = df_train['Ticket'].apply(lambda x: str(int(x.split(' ')[-1])//10))

    # Apply al separations for test data
    df_test['Ticket'] = df_test['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())
    df_test['TicketPrefix'] = df_test['Ticket'].apply(lambda x: get_prefix(x))
    df_test['Ticket_Number'] = df_test['Ticket'].apply(lambda x: int(x.split(' ')[-1])//1)
    df_test['Ticket_Length'] = df_test['Ticket_Number'].apply(lambda x : len(str(x)))
    df_test['Ticket_FirstDigit'] = df_test['Ticket_Number'].apply(lambda x : int(str(x)[0]))
    df_test['Ticket_Group'] = df_test['Ticket'].apply(lambda x: str(int(x.split(' ')[-1])//10))

    # Drop Ticket, Ticket_Number, Ticket_Length
    df_train = df_train.drop(columns=['Ticket', 'Ticket_Number', 'Ticket_Length'])
    df_test = df_test.drop(columns=['Ticket', 'Ticket_Number', 'Ticket_Length'])

    ticket_table = pd.crosstab(df_train['Pclass'],df_train['Ticket_FirstDigit'],margins=True)
    print(ticket_table)

    # Print the unique 'Prefix' in df_train
    #print(df_train['TicketPrefix'].unique())

    # Count the frequency of each 'Prefix' in df_train
    #print(df_train['TicketPrefix'].value_counts())

    return df_train, df_test

def get_prefix(ticket):
    lead = ticket.split(' ')[0][0]
    if lead.isalpha():
        return ticket.split(' ')[0]
    else:
        return 'NoPrefix'
    
def data_transformation(df_train, df_test):
    complete_ages(df_train, df_test)

    group_size_col(df_train, df_test)

    age_group_sex_col(df_train, df_test)

    ticket_info_col(df_train, df_test)

    return df_train, df_test

