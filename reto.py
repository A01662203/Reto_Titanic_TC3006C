import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Open test.csv and count the number of rows
df_test = pd.read_csv('./data/test.csv')
print(df_test.shape[0])
# Open train.csv and count the number of rows
df_train = pd.read_csv('./data/train.csv')
print(df_train.shape[0])

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

# Get missing percentage per column in df_train
missing_values = df_train.isnull().mean() * 100
print(missing_values)
# Plot the missing values of columns with more than 0% missing values
plt.figure(figsize=(10, 5))
sns.barplot(x=missing_values[missing_values > 0].index, y=missing_values[missing_values > 0])
plt.xticks(rotation=45)
plt.ylabel('Porcentaje')
plt.title('Columnas con valores faltantes en df_train')
plt.show()

print('*'*50)

# Get missing percentage per column in df_test
missing_values_test = df_test.isnull().mean() * 100
# Plot the missing values of columns with more than 0% missing values
plt.figure(figsize=(10, 5))
sns.barplot(x=missing_values_test[missing_values_test > 0].index, y=missing_values_test[missing_values_test > 0])
plt.xticks(rotation=45)
plt.ylabel('Porcentaje')
plt.title('Columnas con valores faltantes en df_test')
plt.show()

print('*'*50)

# Exploratory Data Analysis
df_train_analisis = df_train.copy()

# Generate bins for 'Fare'
df_train_analisis['FareBin'] = pd.qcut(df_train_analisis['Fare'], 4)

# Plot Survival Rate by FareBin
plt.figure(figsize=(10, 5))
sns.barplot(x='FareBin', y='Survived', data=df_train_analisis)
plt.xticks(rotation=45)
plt.ylabel('Survival Rate')
plt.title('Survival Rate by FareBin')
plt.show()

# Generate bins for 'Age'
df_train_analisis['AgeBin'] = pd.cut(df_train_analisis['Age'], 5)

# Plot Survival Rate by AgeBin
plt.figure(figsize=(10, 5))
sns.barplot(x='AgeBin', y='Survived', data=df_train_analisis)
plt.xticks(rotation=45)
plt.ylabel('Survival Rate')
plt.title('Survival Rate by AgeBin')
plt.show()

# Create a new feature combining 'AgeBin' and 'Sex'
df_train_analisis['AgeGroup_Sex'] = df_train_analisis['AgeBin'].astype(str) + '_' + df_train_analisis['Sex']

# Plot Survival Rate by AgeGroup_Sex
plt.figure(figsize=(10, 5))
sns.barplot(x='AgeGroup_Sex', y='Survived', data=df_train_analisis)
plt.xticks(rotation=45)
plt.ylabel('Survival Rate')
plt.title('Survival Rate by AgeGroup_Sex')
plt.show()

# Plot Survival Rate characteristics of Male passengers
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

# Filtrar el DataFrame dejando solo la columna 'AgeGroup_Sex'
df_train_analisis = df_train[['AgeGroup_Sex']]

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Probabilidad de supervivencia por grupo de edad y sexo')

# Crear la tabla dinámica con los conteos para cada estado de supervivencia
pivot = df_train.pivot_table(index='AgeGroup_Sex', columns='Survived', aggfunc='size', fill_value=0)

# Calcular la tasa de supervivencia
pivot['Survival Rate'] = pivot[1] / (pivot[0] + pivot[1])

# Ordenar la tabla dinámica por la tasa de supervivencia en orden ascendente
#pivot = pivot.sort_values(by 'Survival Rate', ascending=True)

# Graficar la tasa de supervivencia
pivot['Survival Rate'].plot(kind='bar', ax=ax)#, color='skyblue')

# Configurar el título y etiquetas
#ax.set_title('Survival Rate by SexGroupAge')
ax.set_ylabel('Probabilidad de supervivencia')
ax.set_ylim(0, 1)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Mostrar la gráfica
plt.tight_layout()
plt.show()