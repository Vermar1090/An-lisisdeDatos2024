import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('titanic-dataset.csv')
print(df.describe())
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Distribución de Edad')
plt.show()
sns.countplot(data=df, x='Survived')
plt.title('Distribución de Supervivientes')
plt.show()
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()
